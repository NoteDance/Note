""" DiffGrad
Implements diffGrad algorithm. It is modified from the pytorch implementation of Adam.
It has been proposed in `diffGrad: An Optimization Method for Convolutional Neural Networks`_.
https://arxiv.org/abs/1909.11015

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class DiffGrad(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="diffgrad",
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

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.previous_grad = []
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
            self.previous_grad.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="previous_grad"
                )
            )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError('diffGrad does not support sparse gradients, please consider SparseAdam instead')
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        previous_grad = self.previous_grad[self._get_variable_index(variable)]
        
        self.step[self._get_variable_index(variable)] += 1

        if self.weight_decay != 0:
            gradient += variable * self.weight_decay
 
        # Decay the first and second moment running average coefficient
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1 - self.beta1))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + (1 - self.beta2) * tf.square(gradient))
        denom = tf.sqrt(exp_avg_sq) + self.epsilon
 
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]
 
        # compute diffgrad coefficient (dfc)
        diff = tf.abs(previous_grad - gradient)
        dfc = 1. / (1. + tf.exp(-diff))
        #self.previous_grad[self._get_variable_index(variable)] = gradient %used in paper but has the bug that previous grad is overwritten with grad and diff becomes always zero. Fixed in the next line.
        self.previous_grad[self._get_variable_index(variable)] = gradient
 			
 		# update momentum with dfc
        exp_avg1 = exp_avg * dfc
 
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
 
        variable.assign_add(-step_size * (exp_avg1 / denom))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass