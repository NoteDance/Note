"""
AdamP Optimizer Implementation copied from https://github.com/clovaai/AdamP/blob/master/adamp/adamp.py

Paper: `Slowing Down the Weight Norm Increase in Momentum-based Optimizers` - https://arxiv.org/abs/2006.08217
Code: https://github.com/clovaai/AdamP

Copyright (c) 2024-present NoteDance.
Apache-2.0 license
"""

import tensorflow as tf
import numpy as np
from keras.src.optimizers import optimizer
import math


def clip(x, min):
    x_dtype = x.dtype
    if x_dtype == tf.int32:
        max = np.iinfo(np.int32).max - 2**7
    elif x_dtype == tf.int64:
        max = np.iinfo(np.int64).max - 2**39
    elif x_dtype == tf.float16:
        max = float(np.finfo(np.float16).max)
    else:
        max = float(np.finfo(np.float32).max)

    return tf.clip_by_value(x, min, max)


def cosine_similarity(x1, x2, axis=1, eps=1e-8):
    w12 = tf.reduce_sum(tf.multiply(x1, x2), axis=axis)
    w1 = tf.reduce_sum(tf.multiply(x1, x1), axis=axis)
    w2 = tf.reduce_sum(tf.multiply(x2, x2), axis=axis)
    n12 = tf.sqrt(clip(w1 * w2, eps * eps))
    cos_sim = w12 / n12
    return cos_sim


def _channel_view(x):
    return tf.reshape(x, (x.shape[0], -1))


def _layer_view(x):
    return tf.reshape(x, (1, -1))


def projection(p, grad, perturb, delta: float, wd_ratio: float, eps: float):
    wd = 1.
    expand_size = (-1,) + (1,) * (len(p.shape) - 1)
    for view_func in [_channel_view, _layer_view]:
        param_view = view_func(p)
        grad_view = view_func(grad)
        cosine_sim = tf.abs(cosine_similarity(grad_view, param_view, axis=1, eps=eps))

        # FIXME this is a problem for PyTorch XLA
        if tf.reduce_max(cosine_sim) < delta / math.sqrt(param_view.shape[1]):
            p_n = p / tf.reshape(tf.norm(param_view, ord=2, axis=1) + eps, (expand_size))
            perturb -= p_n * tf.reshape(tf.reduce_sum(view_func(p_n * perturb), axis=1), (expand_size))
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


class AdamP(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0,
        delta=0.1,
        wd_ratio=0.1,
        nesterov=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamp",
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
        self.delta = delta
        self.wd_ratio = wd_ratio
        self.nesterov = nesterov

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.step = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        exp_avg, exp_avg_sq = self.exp_avg[self._get_variable_index(variable)], self.exp_avg_sq[self._get_variable_index(variable)]
        beta1, beta2 = self.beta1, self.beta2

        self.step[self._get_variable_index(variable)] += 1
        bias_correction1 = 1 - beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - beta2 ** self.step[self._get_variable_index(variable)]

        exp_avg.assign(exp_avg * beta1 + gradient * (1 - beta1))
        exp_avg_sq.assign(exp_avg_sq * beta2 + gradient * gradient * (1 - beta2))

        denom = (tf.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + self.epsilon
        step_size = lr / bias_correction1

        if self.nesterov:
            perturb = (beta1 * exp_avg + (1 - beta1) * gradient) / denom
        else:
            perturb = exp_avg / denom

        # Projection
        wd_ratio = 1.
        if len(variable.shape) > 1:
            perturb, wd_ratio = projection(variable, gradient, perturb, self.delta, self.wd_ratio, self.epsilon)
        
        # Weight decay
        if self.weight_decay > 0:
            variable.assign(variable * (1. - lr * self.weight_decay * wd_ratio))
        
        # Step
        variable.assign(variable + (perturb * -step_size))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "wd_ratio": self.wd_ratio,
                "nesterov": self.nesterov,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass