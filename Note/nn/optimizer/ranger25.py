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
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )
            self.exp_avg_slow[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_slow"
                                                    )
            self.slow_momentum[self._get_variable_index(var)].assign(var)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.exp_avg_slow = []
        self.slow_momentum = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.exp_avg_slow.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_slow"
                                                    ))
            self.slow_momentum.append(tf.Variable(var))
            self._track_variable(self.slow_momentum[-1])
    
    @staticmethod
    def schedule_alpha(t_alpha_beta3, step, alpha):
        return alpha if t_alpha_beta3 is None else tf.minimum(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3, step, beta1, beta3):
        if t_alpha_beta3 is None:
            return beta3

        log_beta1, log_beta3 = math.log(beta1), math.log(beta3)

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
                    'AdaGC does not support sparse gradients')
            
            lr = tf.cast(learning_rate, p.dtype)
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2_sq = tf.sqrt(1 - beta2 ** step)
            
            step_size = lr / bias_correction1
            clip = tf.pow(step, 0.25)
            
            alpha_t = self.schedule_alpha(self.t_alpha_beta3, step, self.alpha)
            beta3_t = self.schedule_beta3(self.t_alpha_beta3, step, beta1, beta3)
            
            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * self.weight_decay
                
            grads[self._get_variable_index(p)] = agc(p, g)

            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            exp_avg_slow = self.exp_avg_slow[self._get_variable_index(p)]
                
            normed_grad = tf.clip_by_value(
                g / tf.maximum(tf.sqrt(exp_avg_sq), self.epsilon if self.epsilon is not None else 1e-8),
                clip_value_min=-clip,
                clip_value_max= clip,
            )
            
            exp_avg.assign(exp_avg * beta1 + normed_grad * (1.0 - beta1))
            exp_avg_sq.assign(exp_avg_sq * beta2 + g * g * (1.0 - beta2))
            exp_avg_slow.assign(exp_avg_slow * beta3_t + normed_grad * (1.0 - beta3_t))
            
            update = exp_avg
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
            
            de_nom = tf.sqrt(exp_avg_sq) / bias_correction2_sq
            
            if self.epsilon is not None:
                p.assign_add(-step_size * update / de_nom + self.epsilon)
            else:
                p.assign_add(tf.atan2(update, de_nom) * -step_size)
            
            def true_fn():
                slow_p = self.slow_momentum[self._get_variable_index(p)]
                slow_p.assign(slow_p + self.lookahead_blending_alpha * (p - slow_p))
                p.assign(slow_p)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
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
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass