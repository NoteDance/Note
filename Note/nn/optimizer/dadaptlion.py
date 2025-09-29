""" DAdaptLion
https://arxiv.org/abs/2301.07733

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


class DAdaptLion(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        d0=1e-6,
        weight_decouple=True,
        fixed_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="dadaptlion",
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
        self.d0 = d0
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self.numerator_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.numerator_accumulator = tf.Variable(0.0)
        self.d0_ = tf.Variable(self.d0)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.numerator_accumulator)
        self._track_variable(self.d0_)
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="s"
                                                    )
            
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.s = []
        self.numerator_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.numerator_accumulator = tf.Variable(0.0)
        self.d0_ = tf.Variable(self.d0)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.numerator_accumulator)
        self._track_variable(self.d0_)
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
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
        beta2_sq = math.sqrt(self.beta2)
        
        d_lr = self.d0_ * learning_rate
        
        for variable, grad in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(grad):
                raise RuntimeError(
                    'DAdaptLion does not support sparse gradients')
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else d_lr)))
            elif self.weight_decay > 0.0:
                grad += variable * self.weight_decay
            
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            s = self.s[self._get_variable_index(variable)]
            
            d_lr = tf.cast(d_lr, variable.dtype)
            
            update = tf.math.sign(exp_avg * self.beta1 + grad * (1.0 - self.beta1))
            variable.assign_add(update * -d_lr)
            
            exp_avg.assign(exp_avg * self.beta2 + grad * (1.0 - self.beta2) * d_lr)
            
            self.numerator_accumulator.assign_add(tf.cast(tf.tensordot(tf.reshape(update, [-1]), tf.reshape(s, [-1]), axes=1) * d_lr, tf.float32))
            s.assign(s * beta2_sq + update * (1.0 - beta2_sq) * d_lr)
            
            self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
        
        self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_accumulator * (1.0 - beta2_sq))
        
        def update_fn():
            d = self.d0_
            if self.lr > 0.0:
                d_hat = self.numerator_weighted / ((1.0 - beta2_sq) * self.sk_l1)
                d = tf.maximum(self.d0_, d_hat)
            
            self.d0_.assign(d)
            
        def no_update_fn():
            pass
        
        tf.cond(self.sk_l1 == 0, no_update_fn, update_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "d0": self.d0,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class DAdaptLion_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        d0=1e-6,
        weight_decouple=True,
        fixed_decay=False,
        orthograd=True,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        lookahead=True,
        pnm=True,
        agc=True,
        cautious=True,
        subset_size=-1,
        sn=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="dadaptlion_e",
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
        self.d0 = d0
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.orthograd = orthograd
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.agc = agc
        self.cautious = cautious
        self.subset_size = subset_size
        self.sn = sn
            
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.s = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []
        self.subset_size_ = []
        self.numerator_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.numerator_accumulator = tf.Variable(0.0)
        self.d0_ = tf.Variable(self.d0)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.numerator_accumulator)
        self._track_variable(self.d0_)
        for var in var_list:
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(var))
                self._track_variable(self.slow_momentum[-1])
            if self.pnm:
                self.pos_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="pos_momentum"
                    )
                )
                self.neg_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="neg_momentum"
                    )
                )
            else:
                self.exp_avg.append(self.add_variable_from_reference(
                                    reference_variable=var, name="exp_avg"
                                                        ))
            self.s.append(self.add_variable_from_reference(
                                reference_variable=var, name="s"
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
    
    def apply_orthogonal_gradients(self, params, grads, eps = 1e-16):
        for p, g in zip(params, grads):
            if tf.keras.backend.is_sparse(g):
                continue
            
            original_shape = g.shape
            w = tf.reshape(p, [-1])
            g = tf.reshape(g, [-1])
    
            proj = tf.tensordot(w, g, axes=1) / (tf.tensordot(w, w, axes=1) + eps)
            g_ortho = tf.cast(g, tf.float32) - proj * w
            g_norm = tf.norm(g)
            g_ortho_norm = tf.norm(g_ortho)
            g_ortho_scaled = g_ortho * (g_norm / (g_ortho_norm + eps))
            
            grads[self._get_variable_index(p)] = tf.reshape(g_ortho_scaled, original_shape)
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.orthograd:
            self.apply_orthogonal_gradients(trainable_variables, grads)
            
        beta2_sq = math.sqrt(self.beta2)
        
        d_lr = self.d0_ * learning_rate
        
        for variable, grad in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(grad):
                raise RuntimeError(
                    'DAdaptLion_e does not support sparse gradients')
            
            step = tf.cast(self.iterations + 1, variable.dtype)
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else d_lr)))
            elif self.weight_decay > 0.0:
                grad += variable * self.weight_decay
            
            if self.agc:
                grads[self._get_variable_index(variable)] = agc(variable, grad)  
                grad = grads[self._get_variable_index(variable)]
            
            size = tf.size(grad)
            
            if not self.pnm:
                exp_avg = self.exp_avg[self._get_variable_index(variable)]
            s = self.s[self._get_variable_index(variable)]
            
            if not self.pnm:
                exp_avg.assign(exp_avg * self.beta1 + grad * (1.0 - self.beta1))
            else:
                noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
                def true_fn():
                    return self.pos_momentum[self._get_variable_index(variable)], self.neg_momentum[self._get_variable_index(variable)]
                def false_fn():
                    return self.neg_momentum[self._get_variable_index(variable)], self.pos_momentum[self._get_variable_index(variable)]
                pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
                pos_momentum.assign(pos_momentum * self.beta1 ** 2 + grad * (1.0 - self.beta1 ** 2))
                exp_avg = (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)
            
            d_lr = tf.cast(d_lr, variable.dtype)
            
            update = tf.math.sign(exp_avg * self.beta1 + grad * (1.0 - self.beta1))
            
            if self.cautious:
                mask = tf.cast(tf.math.greater(update * grad, 0), grad.dtype)
                numel = tf.cast(tf.size(mask), grad.dtype)
                factor = numel / (tf.reduce_sum(mask) + 1)
                mask = mask * factor
                update = update * mask
                
            variable.assign_add(update * -d_lr)
            
            if self.lookahead:
                def true_fn():
                    slow_p = self.slow_momentum[self._get_variable_index(variable)]
                    slow_p.assign(slow_p + self.lookahead_blending_alpha * (variable - slow_p))
                    variable.assign(slow_p)
                
                def false_fn():
                    pass
            
                tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
            
            exp_avg.assign(exp_avg * self.beta2 + grad * (1.0 - self.beta2) * d_lr)
            
            if self.sn:
                reshaped_update = tf.reshape(update, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
                reshaped_s = tf.reshape(s, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
                self.numerator_accumulator.assign_add(tf.cast(tf.tensordot(reshaped_update, reshaped_s, axes=len(reshaped_update.shape)) * d_lr, tf.float32))
            else:
                self.numerator_accumulator.assign_add(tf.cast(tf.tensordot(tf.reshape(update, [-1]), tf.reshape(s, [-1]), axes=1) * d_lr, tf.float32))
            s.assign(s * beta2_sq + update * (1.0 - beta2_sq) * d_lr)
            
            self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
        
        self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_accumulator * (1.0 - beta2_sq))
        
        def update_fn():
            d = self.d0_
            if self.lr > 0.0:
                d_hat = self.numerator_weighted / ((1.0 - beta2_sq) * self.sk_l1)
                d = tf.maximum(self.d0_, d_hat)
            
            self.d0_.assign(d)
            
        def no_update_fn():
            pass
        
        tf.cond(self.sk_l1 == 0, no_update_fn, update_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "d0": self.d0,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "orthograd": self.orthograd,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "agc": self.agc,
                "cautious": self.cautious,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "subset_size_": self.subset_size_,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
