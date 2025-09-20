""" Ranger21_e
Integrating the latest deep learning components into a single optimizer.

Here's the components
    * uses the AdamW optimizer as its core (or, optionally, MadGrad)
    * Adaptive gradient clipping
    * Gradient centralization
    * Positive-Negative momentum
    * Norm loss
    * Stable weight decay
    * Linear learning rate warm-up
    * Explore-exploit learning rate schedule
    * Lookahead
    * Softplus transformation
    * Gradient Normalization
    * Corrects the denominator (AdamD).
    * Subset-based second-moment estimation (subset normalization)
    * D-Adaptation

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


def unit_norm(x, ord = 2.0):
    r"""Get norm of unit."""
    keepdims = True
    axis = None

    x_len = len(x.shape)
    if x_len <= 1:
        keepdims = False
    elif x_len in (2, 3):
        axis = 1
    elif x_len == 4:
        axis = (1, 2, 3)
    else:
        axis = tuple(range(1, x_len))

    return tf.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def agc(
    p, grad, agc_eps = 1e-3, agc_clip_val = 1e-2, eps = 1e-6
):
    r"""Clip gradient values in excess of the unit wise norm."""
    max_norm = tf.maximum(unit_norm(p), agc_eps) * agc_clip_val
    g_norm = tf.maximum(unit_norm(grad), eps)

    clipped_grad = grad * (max_norm / g_norm)

    return tf.where(g_norm > max_norm, clipped_grad, grad)


def softplus(x, beta=1.0, threshold=20.0):
    if beta != 1.0:
        x = x * beta
    x = tf.where(
        x > threshold,
        x,
        tf.math.log(1 + tf.exp(x))
    )
    if beta != 1.0:
        x = x / beta
    return x


def closest_smaller_divisor_of_n_to_k(n, k):
    r"""Get closest smaller divisor of n to k."""
    def true_fn():
        return k
    
    def false_fn():
        def true_fn():
            raise ValueError
        def false_fn():
            pass
        tf.cond(tf.logical_or(n <= 1, k <= 1), true_fn, false_fn)
        closest_smaller_divisor = -7
        for i in tf.range(k, 0, -1):
            def true_fn():
                def true_fn():
                    return i
                def false_fn():
                    return -7
                return tf.cond(closest_smaller_divisor == -7, true_fn, false_fn)
            def false_fn():
                return -7  # pragma: no cover
            closest_smaller_divisor = tf.cond(n % i == 0, true_fn, false_fn)
        return closest_smaller_divisor
    
    closest_smaller_divisor = tf.cond(n % k == 0, true_fn, false_fn)
    
    def true_fn():
        return -1
    def false_fn():
        return closest_smaller_divisor
    closest_smaller_divisor = tf.cond(closest_smaller_divisor == -7, true_fn, false_fn)
    
    return closest_smaller_divisor


class Ranger21_e(optimizer.Optimizer):
    def __init__(
        self,
        num_iterations,
        learning_rate=1e-3,
        epsilon=1e-8,
        weight_decay=1e-4,
        beta0=0.9,
        betas=(0.9, 0.999),
        use_softplus=True,
        beta_softplus=50.0,
        disable_lr_scheduler=False,
        num_warm_up_iterations=None,
        num_warm_down_iterations=None,
        warm_down_min_lr=3e-5,
        agc_clipping_value=1e-2,
        agc_eps=1e-3,
        centralize_gradients=True,
        normalize_gradients=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        weight_decouple=True,
        fixed_decay=False,
        norm_loss_factor=1e-4,
        adam_debias=False,
        subset_size=-1,
        sn=True,
        d0=1e-6,
        growth_rate=float('inf'),
        DAdapt=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="ranger21_e",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.beta0 = beta0
        self.betas = betas
        self.epsilon = epsilon
        self.min_lr = warm_down_min_lr
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.disable_lr_scheduler = disable_lr_scheduler
        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.norm_loss_factor = norm_loss_factor
        self.adam_debias = adam_debias
        self.subset_size = subset_size
        self.sn = sn
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt

        self.starting_lr: float = learning_rate
        self.current_lr: float = learning_rate

        self.num_warm_up_iterations: int = (
            self.build_warm_up_iterations(num_iterations, betas[1])
            if num_warm_up_iterations is None
            else num_warm_up_iterations
        )
        self.num_warm_down_iterations: int = (
            self.build_warm_down_iterations(num_iterations)
            if num_warm_down_iterations is None
            else num_warm_down_iterations
        )
        self.start_warm_down: int = num_iterations - self.num_warm_down_iterations
        self.warm_down_lr_delta: float = self.starting_lr - self.min_lr
    
    @staticmethod
    def build_warm_up_iterations(total_iterations, beta2, warm_up_pct = 0.22):
        warm_up_iterations = math.ceil(2.0 / (1.0 - beta2))  # default un-tuned linear warmup
        beta_pct = warm_up_iterations / total_iterations
        if beta_pct > 0.45:
            return warm_up_pct * total_iterations
        else:
            return warm_up_iterations

    @staticmethod
    def build_warm_down_iterations(total_iterations, warm_down_pct = 0.72):
        start_warm_down = warm_down_pct * total_iterations
        return total_iterations - start_warm_down

    def warm_up_dampening(self, lr, step):
        def true_fn():
            return lr
        
        def false_fn():
            warm_up_current_pct = tf.minimum(1.0, (step / self.num_warm_up_iterations))

            self.current_lr = lr * warm_up_current_pct
            
            return self.current_lr
        
        return tf.cond(step > self.num_warm_up_iterations, true_fn, false_fn)

    def warm_down(self, lr, iteration):
        def true_fn():
            return lr
        
        def false_fn():
            warm_down_iteration = tf.maximum((iteration + 1) - self.start_warm_down, 1)
            warm_down_pct = tf.minimum(warm_down_iteration / (self.num_warm_down_iterations + 1), 1.0)

            self.current_lr = tf.maximum(self.starting_lr - self.warm_down_lr_delta * warm_down_pct, self.min_lr)

            return self.current_lr
        
        return tf.cond(iteration < self.start_warm_down, true_fn, false_fn)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.grad_ma = []
        self.variance_ma = []
        self.lookahead_params = []
        self.neg_grad_ma = []
        self.max_variance_ma = []
        self.param_size = tf.Variable(0)
        self.variance_ma_sum = tf.Variable(1.0)
        self.lookahead_step = tf.Variable(0)
        self._track_variable(self.lookahead_step)
        self.subset_size_ = []
        if self.DAdapt:
            self.s = []
            self.sk_l1 = tf.Variable(0.0)
            self.numerator_acc = tf.Variable(0.0)
            self.numerator_weighted = tf.Variable(0.0)
            self.d0_ = tf.Variable(self.d0)
            self._track_variable(self.sk_l1)
            self._track_variable(self.numerator_acc)
            self._track_variable(self.numerator_weighted)
            self._track_variable(self.d0_)
        for var in var_list:
            self.grad_ma.append(self.add_variable_from_reference(
                                reference_variable=var, name="grad_ma"
                                                    ))
            if self.sn:
                size = tf.size(var)
                
                def true_fn():
                    return self.subset_size
                def false_fn():
                    return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                    size,
                    tf.cond(self.subset_size > 0, true_fn, false_fn)
                ))
            
                reshaped_grad = tf.reshape(var, (size // self.subset_size_[-1], self.subset_size_[-1]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                second_moment_update = tf.Variable(second_moment_update)
                self.variance_ma.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="variance_ma"
                    ))
            else:
                self.variance_ma.append(self.add_variable_from_reference(
                        reference_variable=var, name="variance_ma"
                    ))
            self.lookahead_params.append(tf.Variable(var))
            self._track_variable(self.lookahead_params[-1])
            self.neg_grad_ma.append(self.add_variable_from_reference(
                                reference_variable=var, name="neg_grad_ma"
                                                    ))
            self.max_variance_ma.append(self.add_variable_from_reference(
                                reference_variable=var, name="max_variance_ma"
                                                    ))
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                                    reference_variable=var, name="s"
                                                        ))
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        self.param_size.assign(0)
        self.variance_ma_sum.assign(1.0)
        beta1, beta2 = self.betas
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'Ranger21_e does not support sparse gradients')
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction2 = 1 - beta2 ** step
            
            size = tf.size(g)
            
            self.param_size.assign_add(tf.size(p))
            
            # Apply Adaptive Gradient Clipping (AGC)
            grads[self._get_variable_index(p)] = agc(p, g, self.agc_eps, self.agc_clipping_value)
            
            # Apply gradient centralization & normalization
            size_ = len(g.shape)
            if size_ > 1:
                grads[self._get_variable_index(p)] += tf.reduce_mean(-g, axis=tuple(range(1, size)), keepdims=True)
            def true_fn():
                s = tf.math.reduce_std(grads[self._get_variable_index(p)]) + 1e-8
                grads[self._get_variable_index(p)] = grads[self._get_variable_index(p)] / s
            def false_fn():
                pass
            tf.cond(tf.size(g) > 2, true_fn, false_fn)
            
            g = grads[self._get_variable_index(p)]
            
            # second moment estimation
            # using positive-negative momentum and bias correction
            variance_ma = self.variance_ma[self._get_variable_index(p)]
            if self.sn:
                reshaped_grad = tf.reshape(g, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
            else:
                second_moment_update = tf.pow(g, 2)
            variance_ma.assign(variance_ma * beta2 + second_moment_update * (1.0 - beta2))
            self.variance_ma_sum.assign_add(tf.reduce_sum((variance_ma / bias_correction2)))
        
        variance_normalized = tf.sqrt(self.variance_ma_sum / self.param_size)
        
        # Phase 2 - Apply weight decay and step
        for p, g in zip(trainable_variables, grads):
            lr = tf.cast(learning_rate, p.dtype)
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - beta2 ** step
            bias_correction2_sq = tf.sqrt(1 - beta2 ** step)
            
            d_lr = self.d0_ * lr if self.adam_debias else self.d0_ * lr / bias_correction1
            d_lr = tf.cast(d_lr, p.dtype)
    
            noise_norm = math.sqrt((1.0 + beta2) ** 2 + beta2 ** 2)  # fmt: skip
    
            if not self.DAdapt:
                # warm up & down
                if self.disable_lr_scheduler:
                    lr = lr
                else:
                    lr = self.warm_up_dampening(lr, step)
                    lr = self.warm_down(lr, step)

            # stable weight decay
            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else self.lr) * 1.0 / variance_normalized))

            # norm loss
            correction = 2.0 * self.norm_loss_factor * (1.0 - 1.0 / unit_norm(p) + self.epsilon)
            if not self.DAdapt:
                p.assign(p * (1.0 - lr * correction))
            else:
                p.assign(p * (1.0 - d_lr * correction))

            def true_fn():
                return self.grad_ma[self._get_variable_index(p)], self.neg_grad_ma[self._get_variable_index(p)]
            def false_fn():
                return self.neg_grad_ma[self._get_variable_index(p)], self.grad_ma[self._get_variable_index(p)]
            grad_ma, neg_grad_ma = tf.cond(step % 2 == 1, true_fn, false_fn)

            variance_ma = self.variance_ma[self._get_variable_index(p)]
            variance_ma.assign(tf.maximum(self.max_variance_ma[self._get_variable_index(p)], variance_ma))

            de_nom = (tf.sqrt(variance_ma) / bias_correction2_sq) + self.epsilon

            if self.use_softplus:
                de_nom = softplus(de_nom, beta=self.beta_softplus)
            
            if self.DAdapt:
                s = self.s[self._get_variable_index(p)]
            
                flat_grad = tf.reshape(g, [-1])
                flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                self.numerator_acc.assign_add(tf.cast(lr * dot_val, tf.float32))

            size = len(g.shape)
            if size > 1:
                grads[self._get_variable_index(p)] += tf.reduce_mean(-g, axis=tuple(range(1, size)), keepdims=True)
            def true_fn():
                s = tf.math.reduce_std(grads[self._get_variable_index(p)]) + 1e-8
                grads[self._get_variable_index(p)] = grads[self._get_variable_index(p)] / s
            def false_fn():
                pass
            tf.cond(tf.size(g) > 2, true_fn, false_fn)
            
            g = grads[self._get_variable_index(p)]

            if self.DAdapt:
                beta2_sq = math.sqrt(beta2)
                grad_ma.assign((beta1 ** 2) * grad_ma + d_lr * (1 - beta1 ** 2) * g)
                s.assign(s * beta2_sq + g * d_lr * (1.0 - beta2_sq))
                self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
            else:
                grad_ma.assign((beta1 ** 2) * grad_ma + (1 - beta1 ** 2) * g)
            
            if not self.DAdapt:
                step_size = lr if self.adam_debias else lr / bias_correction1
    
                pn_momentum = (grad_ma * 2.0 + neg_grad_ma * -1.0) * (1.0 / noise_norm)
                if self.sn:
                    numerator = tf.reshape(pn_momentum, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    normed_grad = tf.reshape((numerator / de_nom), p.shape)
                    update = -step_size * normed_grad
                else:
                    update = -step_size * pn_momentum / de_nom
                p.assign_add(update)
        
        if not self.DAdapt:
            self.lookahead_process_step()
        
        if self.DAdapt:
            def update_fn():
                d_lr = self.d0_ * learning_rate
                
                beta2_sq = math.sqrt(self.beta2)
                
                d = self.d0_
                self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                
                if self.lr > 0.0:
                    d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                    d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                
                self.d0_.assign(d)
                
                for p in zip(trainable_variables):
                    d_lr = tf.cast(d_lr, p.dtype)
                        
                    step_size = d_lr if self.adam_debias else d_lr / bias_correction1
                    
                    def true_fn():
                        return self.grad_ma[self._get_variable_index(p)], self.neg_grad_ma[self._get_variable_index(p)]
                    def false_fn():
                        return self.neg_grad_ma[self._get_variable_index(p)], self.grad_ma[self._get_variable_index(p)]
                    grad_ma, neg_grad_ma = tf.cond(step % 2 == 1, true_fn, false_fn)
        
                    variance_ma = self.variance_ma[self._get_variable_index(p)]
                    variance_ma.assign(tf.maximum(self.max_variance_ma[self._get_variable_index(p)], variance_ma))
        
                    de_nom = (tf.sqrt(variance_ma) / bias_correction2_sq) + self.epsilon
        
                    if self.use_softplus:
                        de_nom = softplus(de_nom, beta=self.beta_softplus)
        
                    pn_momentum = (grad_ma * 2.0 + neg_grad_ma * -1.0) * (1.0 / noise_norm)
                    if self.sn:
                        numerator = tf.reshape(pn_momentum, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        normed_grad = tf.reshape((numerator / de_nom), p.shape)
                        update = -step_size * normed_grad
                    else:
                        update = -step_size * pn_momentum / de_nom
                    p.assign_add(update)
                
                self.lookahead_process_step()
            
            def no_update_fn():
                pass
    
            tf.cond(self.sk_l1 == 0, no_update_fn, update_fn)
    
    def lookahead_process_step(self):
        self.lookahead_step.assign_add(1)
        
        def true_fn():
            self.lookahead_step.assign(0)
            for p in self._trainable_variables:
                p.assign(p * self.lookahead_blending_alpha + self.lookahead_params[self._get_variable_index(p)] * (1.0 - self.lookahead_blending_alpha))
                self.lookahead_params[self._get_variable_index(p)].assign(p)
        
        def false_fn():
            pass
        
        tf.cond(self.lookahead_step >= self.lookahead_merge_time, true_fn, false_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "num_iterations": self.num_iterations,
                "beta0": self.beta0,
                "betas": self.betas,
                "epsilon": self.epsilon,
                "warm_down_min_lr": self.warm_down_min_lr,
                "use_softplus": self.use_softplus,
                "beta_softplus": self.beta_softplus,
                "disable_lr_scheduler": self.disable_lr_scheduler,
                "agc_clipping_value": self.agc_clipping_value,
                "agc_eps": self.agc_eps,
                "centralize_gradients": self.centralize_gradients,
                "normalize_gradients": self.normalize_gradients,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "norm_loss_factor": self.norm_loss_factor,
                "adam_debias": self.adam_debias,
                "starting_lr": self.starting_lr,
                "current_lr": self.current_lr,
                "num_warm_up_iterations": self.num_warm_up_iterations,
                "num_warm_down_iterations": self.num_warm_down_iterations,
                "start_warm_down": self.start_warm_down,
                "warm_down_lr_delta": self.warm_down_lr_delta,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "DAdapt": self.DAdapt,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass