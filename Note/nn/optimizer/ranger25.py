""" Ranger25

Mixin' every fancy optimizer hacks.

    Here's the components
        * ADOPT
        * AdEMAMix
        * Cautious
        * StableAdamW or Adam-atan2
        * OrthoGrad
        * Adaptive gradient clipping
        * Lookahead
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


class Ranger25(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.98, 0.9999),
        epsilon=1e-8,
        weight_decay=1e-3,
        alpha=5.0,
        t_alpha_beta3=None,
        lookahead_merge_time=5,
        lookahead_blending_alpha=0.5,
        cautious=True,
        stable_adamw=True,
        orthograd=True,
        weight_decouple=True,
        fixed_decay=False,
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
        name="ranger25",
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
        self.betas = betas
        self.epsilon = epsilon
        self.alpha = alpha
        self.t_alpha_beta3 = t_alpha_beta3
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.cautious = cautious
        self.stable_adamw = stable_adamw
        self.orthograd = orthograd
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.subset_size = subset_size
        self.sn = sn
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.exp_avg_slow = []
        self.slow_momentum = []
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
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_slow.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_slow"
                                                    ))
            self.slow_momentum.append(tf.Variable(var))
            self._track_variable(self.slow_momentum[-1])
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
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="exp_avg_sq"
                    ))
            else:
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=var, name="exp_avg_sq"
                    ))
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                                    reference_variable=var, name="s"
                                                            ))
    
    @staticmethod
    def schedule_alpha(t_alpha_beta3, step, alpha):
        return alpha if t_alpha_beta3 is None else tf.minimum(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3, step, beta1, beta3):
        if t_alpha_beta3 is None:
            return beta3

        log_beta1, log_beta3 = tf.math.log(beta1), tf.math.log(beta3)

        return tf.minimum(
            tf.exp(
                log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)
            ),
            beta3,
        )

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
        beta1, beta2, beta3 = self.betas
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'Ranger25 does not support sparse gradients')
            
            beta1 = tf.cast(beta1, p.dtype)
            beta2 = tf.cast(beta2, p.dtype)
            beta3 = tf.cast(beta3, p.dtype)
            
            if self.DAdapt:
                lr = tf.cast(self.lr, p.dtype)
            else:
                lr = tf.cast(learning_rate, p.dtype)
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2_sq = tf.sqrt(1 - beta2 ** step)
            
            step_size = lr * bias_correction2_sq / bias_correction1
            clip = tf.pow(step, 0.25)
            
            if self.DAdapt:
                # it's not Adam Debias
                d_lr = self.d0 * self.lr * bias_correction2_sq / bias_correction1
            
            alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
            beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
            
            size = tf.size(g)
            
            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * self.weight_decay
                
            grads[self._get_variable_index(p)] = agc(p, g)

            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
            
            if self.sn:
                reshaped_grad = tf.reshape(g, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
            else:
                second_moment_update = tf.pow(g, 2)
            
            exp_avg_sq.assign(exp_avg_sq * beta2 + second_moment_update * (1.0 - beta2))
            
            de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
            
            if self.DAdapt:
                s = self.s[self._get_variable_index(p)]
            
                flat_grad = tf.reshape(g, [-1])
                flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
                dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
                
                d_lr = tf.cast(d_lr, dtype=p.dtype)
            
            if self.sn:
                if self.DAdapt:
                    beta2_sq = math.sqrt(beta2)
                    exp_avg.assign(exp_avg * self.beta1 + g * d_lr * (1.0 - self.beta1))
                    s.assign(s * beta2_sq + g * d_lr * (1.0 - beta2_sq))
                    self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
                else:
                    exp_avg.assign(exp_avg * self.beta1 + g * (1.0 - self.beta1))
                numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                normed_grad = tf.reshape((numerator / de_nom), p.shape)
                update = normed_grad
            else:
                normed_grad = tf.clip_by_value(
                    g / tf.maximum(tf.sqrt(exp_avg_sq), self.epsilon if self.epsilon is not None else 1e-8),
                    clip_value_min=-clip,
                    clip_value_max= clip,
                )
                exp_avg.assign(exp_avg * beta1 + normed_grad * (1.0 - beta1))
                update = exp_avg

            if not self.DAdapt:
                exp_avg_slow.assign(exp_avg_slow * beta3_t + normed_grad * (1.0 - beta3_t))
                
                if self.cautious:
                    mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                    numel = tf.cast(tf.size(mask), g.dtype)
                    factor = numel / (tf.reduce_sum(mask) + 1)
                    mask = mask * factor
                    update = update * mask
                
                if self.stable_adamw:
                    step_size /= tf.clip_by_value(
                                    tf.sqrt(tf.reduce_mean(tf.pow(g, 2) / tf.maximum(exp_avg_sq, self.epsilon))),
                                    clip_value_min=1.0,
                                    clip_value_max=tf.float64.max
                                    )
                    
                update += exp_avg_slow * alpha_t
                
                if self.epsilon is not None:
                    if self.sn:
                        p.assign_add(-step_size * update)
                    else:
                        p.assign_add(-step_size * update / de_nom)
                else:
                    p.assign_add(tf.atan2(update, de_nom) * -step_size)
                
                def true_fn():
                    slow_p = self.slow_momentum[self._get_variable_index(p)]
                    slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                    p.assign(slow_p)
                
                def false_fn():
                    pass
                
                tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
        
        if self.DAdapt:
            def update_fn():
                beta2_sq = math.sqrt(self.beta2)
                
                d = self.d0_
                self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                
                if self.lr > 0.0:
                    d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                    d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                
                self.d0_.assign(d)
                
                for p in zip(trainable_variables):
                    lr = tf.cast(self.lr, p.dtype)
                    
                    alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
                    beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
                    
                    exp_avg = self.exp_avg[self._get_variable_index(p)]
                    exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
                    exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
                    
                    if self.sn:
                        numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                        normed_grad = tf.reshape((numerator / de_nom), p.shape)
                        update = normed_grad
                    else:
                        update = exp_avg
                    
                    exp_avg_slow.assign(exp_avg_slow * beta3_t + normed_grad * (1.0 - beta3_t))
                    
                    if self.cautious:
                        mask = tf.cast(tf.math.greater(update * g, 0), g.dtype)
                        numel = tf.cast(tf.size(mask), g.dtype)
                        factor = numel / (tf.reduce_sum(mask) + 1)
                        mask = mask * factor
                        update = update * mask
                    
                    step_size = lr * bias_correction2_sq / bias_correction1
                    
                    if self.stable_adamw:
                        step_size /= tf.clip_by_value(
                                        tf.sqrt(tf.reduce_mean(tf.pow(g, 2) / tf.maximum(exp_avg_sq, self.epsilon))),
                                        clip_value_min=1.0,
                                        clip_value_max=tf.float64.max
                                        )
                        
                    update += exp_avg_slow * alpha_t
                    
                    if self.epsilon is not None:
                        if self.sn:
                            p.assign_add(-step_size * update)
                        else:
                            p.assign_add(-step_size * update / de_nom)
                    else:
                        p.assign_add(tf.atan2(update, de_nom) * -step_size)
                    
                    def true_fn():
                        slow_p = self.slow_momentum[self._get_variable_index(p)]
                        slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                        p.assign(slow_p)
                    
                    def false_fn():
                        pass
                    
                    tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "betas": self.betas,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "t_alpha_beta3": self.t_alpha_beta3,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "cautious": self.cautious,
                "stable_adamw": self.stable_adamw,
                "orthograd": self.orthograd,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
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