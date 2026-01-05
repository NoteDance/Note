import tensorflow as tf
from keras.src.optimizers import optimizer
import math


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


class Ano(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-4,
        beta1=0.92,
        beta2=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        logarithmic_schedule=False,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="ano",
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.logarithmic_schedule = logarithmic_schedule
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                reference_variable=var, name="exp_avg"
            ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                reference_variable=var, name="exp_avg_sq"
            ))

    def update_step(self, gradient, variable, learning_rate):
        step = tf.cast(self.iterations + 1, variable.dtype)
        lr = tf.cast(learning_rate, variable.dtype)
        
        if self.logarithmic_schedule:
            max_t = tf.maximum(2.0, step)
            beta1 = 1.0 - 1.0 / tf.math.log(max_t)
        else:
            beta1 = self.beta1
        
        bias_correction2 = 1.0 - self.beta2 ** step
        
        if self.maximize:
            gradient = -gradient
        
        idx = self._get_variable_index(variable)
        exp_avg = self.exp_avg[idx]
        exp_avg_sq = self.exp_avg_sq[idx]
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient = gradient + variable * self.weight_decay
        
        exp_avg.assign(exp_avg * beta1 + gradient * (1.0 - beta1))
        
        square_grad = gradient * gradient
        sign_diff = tf.sign(square_grad - exp_avg_sq)
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + sign_diff * square_grad * (1.0 - self.beta2))
        
        de_nom = tf.sqrt(exp_avg_sq / bias_correction2) + self.epsilon
        
        update = tf.abs(gradient) * tf.sign(exp_avg) / de_nom
        variable.assign_add(-lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "logarithmic_schedule": self.logarithmic_schedule,
                "maximize": self.maximize,
            }
        )
        return config
    
    def _apply_weight_decay(self, variables):
        pass


class Ano_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-4,
        beta1=0.92,
        beta2=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        logarithmic_schedule=False,
        subset_size=-1,
        sn=True,
        agc=False,
        gc=False,
        sophia=False,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
        trust_ratio=False,
        trust_clip=False,
        cautious=False,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=False,
        DAdapt=False,
        d0=1e-6,
        growth_rate=float('inf'),
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="ano_e",
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.logarithmic_schedule = logarithmic_schedule
        self.subset_size = subset_size
        self.sn = sn
        self.agc = agc
        self.gc = gc
        self.sophia = sophia
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.cautious = cautious
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.DAdapt = DAdapt
        self.d0 = d0
        self.growth_rate = growth_rate
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.subset_size_ = []
        self.hessian_moment = []
        self.hessian = []
        self.slow_momentum = []
        self.s = []
        if self.DAdapt:
            self.sk_l1 = tf.Variable(0.0)
            self.numerator_acc = tf.Variable(0.0)
            self.numerator_weighted = tf.Variable(0.0)
            self.d0_ = tf.Variable(self.d0)
            self._track_variable(self.sk_l1)
            self._track_variable(self.numerator_acc)
            self._track_variable(self.numerator_weighted)
            self._track_variable(self.d0_)
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                reference_variable=var, name="exp_avg"
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
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)  # fmt: skip
                second_moment_update = tf.Variable(second_moment_update)
                if self.sophia:
                    self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                                reference_variable=second_moment_update, name="hessian"
                                                            )
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="hessian_moment"
                                            ))
                else:
                    self.exp_avg_sq.append(self.add_variable_from_reference(
                            reference_variable=second_moment_update, name="exp_avg_sq"
                        ))
            else:
                if self.sophia:
                    self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                                reference_variable=var, name="hessian"
                                                            )
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=var, name="hessian_moment"
                                            ))
                else:
                    self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=var, name="exp_avg_sq"
                    ))
            
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(var))
                self._track_variable(self.slow_momentum[-1])
            
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                                    reference_variable=var, name="s"
                                                        ))
                
    def compute_hutchinson_hessian(
        self,
        grads,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: str = 'gaussian',
    ) -> None:
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params = [p for p in self._trainable_variables if not tf.keras.backend.is_sparse(p)]
        if len(params) == 0:
            return
        
        grads = [g for g in grads if not tf.keras.backend.is_sparse(g)]

        for i in range(num_samples):
            if distribution == 'rademacher':
                zs = [
                    tf.cast(tf.random.uniform(tf.shape(p), 0, 2, dtype=tf.int32)*2 - 1, p.dtype)
                    for p in params
                ]
            else:
                zs = [tf.random.normal(tf.shape(p), dtype=p.dtype) for p in params]

            h_zs = self.tape.gradient(grads, params, zs)

            for h_z, z, p in zip(h_zs, zs, params):
                size = tf.size(p)
                if self.sn:
                    reshaped_h_z = tf.reshape(h_z, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    reshaped_z = tf.reshape(z, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    hessian_update = tf.reduce_sum(reshaped_h_z * reshaped_z, axis=1, keepdims=True)
                else:
                    hessian_update = h_z * z
                self.hessian[self._get_variable_index(p)].assign_add(hessian_update * alpha / num_samples)
    
    def apply_gradients(self, grads_and_vars, tape):
        self.tape = tape
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.sophia:
            def true_fn1():
                self.compute_hutchinson_hessian(
                    grads,
                    num_samples=self.num_samples,
                    distribution=self.distribution,
                )
            def false_fn1():
                pass
            tf.cond(self.iterations % self.update_period == 0, true_fn1, false_fn1)
            
        for variable, gradient in zip(trainable_variables, grads):
            step = tf.cast(self.iterations + 1, variable.dtype)
            lr = tf.cast(learning_rate, variable.dtype)
            
            if self.logarithmic_schedule:
                max_t = tf.maximum(2.0, step)
                beta1 = 1.0 - 1.0 / tf.math.log(max_t)
            else:
                beta1 = self.beta1
            
            bias_correction2 = 1.0 - self.beta2 ** step
            
            idx = self._get_variable_index(variable)
            
            if self.DAdapt:
                d_lr = self.d0_ * learning_rate / bias_correction2
                d_lr = tf.cast(d_lr, variable.dtype)
                beta2_sq = math.sqrt(self.beta2)
                s = self.s[idx]
            
            if self.agc:
                grads[idx] = agc(variable, gradient)
                gradient = grads[idx]
            
            if self.gc:
                size = len(gradient.shape)
                if size > 1:
                    grads[idx] += tf.reduce_mean(-gradient, axis=tuple(range(1, size)), keepdims=True)
                def true_fn():
                    s = tf.math.reduce_std(grads[idx]) + 1e-8
                    grads[idx] = grads[idx] / s
                def false_fn():
                    pass
                tf.cond(tf.size(gradient) > 2, true_fn, false_fn)
                gradient = grads[idx]
            
            size = tf.size(gradient)
            
            if self.maximize:
                gradient = -gradient
            
            exp_avg = self.exp_avg[idx]
            
            if not self.DAdapt:
                if self.weight_decouple:
                    variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
                elif self.weight_decay > 0.0:
                    gradient = gradient + variable * self.weight_decay
            
            if self.DAdapt:
                exp_avg.assign(exp_avg * beta1 + gradient * d_lr * (1.0 - beta1))
                s.assign(s * beta2_sq + gradient * d_lr * (1.0 - beta2_sq))
                self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
            else:
                exp_avg.assign(exp_avg * beta1 + gradient * (1.0 - beta1))
            
            if self.sophia:
                hessian_moment = self.hessian_moment[idx]
                def true_fn2():
                    hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[idx] * (1.0 - self.beta2))
                def false_fn2():
                    pass
                tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
            else:
                exp_avg_sq = self.exp_avg_sq[idx]
                if self.sn:
                    reshaped_grad = tf.reshape(gradient, (size // self.subset_size_[idx], self.subset_size_[idx]))
                    second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)  # fmt: skip
                else:
                    second_moment_update = tf.pow(gradient, 2)
                sign_diff = tf.sign(second_moment_update - exp_avg_sq)
                exp_avg_sq.assign(exp_avg_sq * self.beta2 + sign_diff * second_moment_update * (1.0 - self.beta2))
            
            if self.sophia:
                de_nom = tf.maximum(hessian_moment, self.epsilon)
            else:
                de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
            
            if self.DAdapt:
                if self.sn:
                    s = tf.reshape(s, (size // self.subset_size_[idx], self.subset_size_[idx]))
                flat_grad = tf.reshape(gradient, [-1])
                flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
            
            if not self.DAdapt:
                if self.sn:
                    exp_avg = tf.reshape(exp_avg, (size // self.subset_size_[idx], self.subset_size_[idx]))
                    gradient = tf.reshape(gradient, (size // self.subset_size_[idx], self.subset_size_[idx]))
                    if self.sophia:
                        de_nom = tf.maximum(hessian_moment, self.epsilon)
                        update = tf.reshape(tf.abs(gradient) * tf.sign(exp_avg) / de_nom, variable.shape)
                        update = tf.clip_by_value(update, clip_value_min=-self.p, clip_value_max=self.p)
                    else:
                        update = tf.reshape(tf.abs(gradient) * tf.sign(exp_avg) / de_nom, variable.shape)
                else:
                    if self.sophia:
                        update = tf.clip_by_value(tf.abs(gradient) * tf.sign(exp_avg) / de_nom, clip_value_min=-self.p, clip_value_max=self.p)
                    else:
                        update = tf.abs(gradient) * tf.sign(exp_avg) / de_nom
            
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    w_norm = tf.norm(variable, ord=2)
                    g_norm = tf.norm(update, ord=2)
                    trust_ratio = w_norm / g_norm
                    trust_ratio = tf.where(
                        w_norm > 0,
                        tf.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if self.trust_clip:
                        trust_ratio = tf.minimum(trust_ratio, 1.0)
                    update *= trust_ratio
                    
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * gradient, 0), gradient.dtype)
                    numel = tf.cast(tf.size(mask), gradient.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
            
                variable.assign_add((-lr / bias_correction2) * update)
            
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[idx]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (variable - slow_p))
                        variable.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
        
        if self.DAdapt:
            def update_fn():
                d_lr = self.d0_ * learning_rate / bias_correction2
                d = self.d0_
                self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                
                if self.lr > 0.0:
                    d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                    d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                
                self.d0_.assign(d)
                
                for variable, gradient in zip(trainable_variables, grads):
                    if self.weight_decouple:
                        variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else d_lr)))
                    elif self.weight_decay > 0.0:
                        gradient = gradient + variable * self.weight_decay
                        
                    d_lr = tf.cast(d_lr, variable.dtype)
                    
                    exp_avg = self.exp_avg[idx]
                    
                    if self.sophia:
                        hessian_moment = self.hessian_moment[idx]
                        def true_fn2():
                            hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[idx] * (1.0 - self.beta2))
                        def false_fn2():
                            pass
                        tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
                    else:
                        exp_avg_sq = self.exp_avg_sq[idx]
                    
                    if self.sophia:
                        de_nom = tf.maximum(hessian_moment, self.epsilon)
                    else:
                        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
                        
                if self.sn:
                    exp_avg = tf.reshape(exp_avg, (size // self.subset_size_[idx], self.subset_size_[idx]))
                    gradient = tf.reshape(gradient, (size // self.subset_size_[idx], self.subset_size_[idx]))
                    if self.sophia:
                        de_nom = tf.maximum(hessian_moment, self.epsilon)
                        update = tf.reshape(tf.abs(gradient) * tf.sign(exp_avg) / de_nom, variable.shape)
                        update = tf.clip_by_value(update, clip_value_min=-self.p, clip_value_max=self.p)
                    else:
                        update = tf.reshape(tf.abs(gradient) * tf.sign(exp_avg) / de_nom, variable.shape)
                else:
                    if self.sophia:
                        update = tf.clip_by_value(tf.abs(gradient) * tf.sign(exp_avg) / de_nom, clip_value_min=-self.p, clip_value_max=self.p)
                    else:
                        update = tf.abs(gradient) * tf.sign(exp_avg) / de_nom
            
                if self.trust_ratio:
                    # Layer-wise LR adaptation
                    w_norm = tf.norm(variable, ord=2)
                    g_norm = tf.norm(update, ord=2)
                    trust_ratio = w_norm / g_norm
                    trust_ratio = tf.where(
                        w_norm > 0,
                        tf.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if self.trust_clip:
                        trust_ratio = tf.minimum(trust_ratio, 1.0)
                    update *= trust_ratio
                    
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * gradient, 0), gradient.dtype)
                    numel = tf.cast(tf.size(mask), gradient.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
            
                variable.assign_add(-1.0 * update)
            
                if self.lookahead:
                    def true_fn():
                        slow_p = self.slow_momentum[idx]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (variable - slow_p))
                        variable.assign(slow_p)
                    
                    def false_fn():
                        pass
                
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
                    

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "logarithmic_schedule": self.logarithmic_schedule,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "agc": self.agc,
                "gc": self.gc,
                "sophia": self.sophia,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
                "cautious": self.cautious,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "DAdapt": self.DAdapt,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "maximize": self.maximize,
            }
        )
        return config
    
    def _apply_weight_decay(self, variables):
        pass