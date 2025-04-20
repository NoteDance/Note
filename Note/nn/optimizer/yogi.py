""" Yogi
Implements Yogi Optimizer Algorithm.
It has been proposed in `Adaptive methods for Nonconvex Optimization`__.
https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class Yogi(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-3,
        weight_decay=0,
        initial_accumulator=1e-6,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="yogi",
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
        self.initial_accumulator = initial_accumulator

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.step = []
        for var in var_list:
            self.exp_avg.append(
                tf.Variable(
                    tf.fill(tf.shape(var), self.initial_accumulator),
                    dtype=var.dtype,
                )
            )
            self._track_variable(self.exp_avg[-1])
            self.exp_avg_sq.append(
                tf.Variable(
                    tf.fill(tf.shape(var), self.initial_accumulator),
                    dtype=var.dtype,
                )
            )
            self._track_variable(self.exp_avg_sq[-1]) 
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                "Yogi does not support sparse gradients, "
                "please consider SparseAdam instead"
            )
            
        exp_avg, exp_avg_sq = (
            self.exp_avg[self._get_variable_index(variable)],
            self.exp_avg_sq[self._get_variable_index(variable)],
        )
        
        self.step[self._get_variable_index(variable)] += 1
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]

        if self.weight_decay != 0:
            gradient = gradient + variable * self.weight_decay
        
        # Decay the first and second moment running average coefficient
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1 - self.beta1))

        grad_squared = gradient * gradient

        exp_avg_sq.assign_add(
            -(1 - self.beta2) * (tf.sign(exp_avg_sq - grad_squared) * grad_squared)
        )

        denom = (tf.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + self.epsilon
        step_size = lr / bias_correction1
        variable.assign_add(-step_size * (exp_avg / denom))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "initial_accumulator": self.initial_accumulator,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass