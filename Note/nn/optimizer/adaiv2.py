""" AdaiV2
Implements AdaiV2.
It is a generalized variant of Adai based on
`Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum`.
https://arxiv.org/abs/2006.15815

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class AdaiV2(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate,
        beta0=0.1,
        beta2=0.99,
        epsilon=1e-03,
        weight_decay=0,
        dampening=1.,
        decoupled=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adaiv2",
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
        self.beta0 = beta0
        self.beta2 = beta2
        self.epsilon = epsilon
        self.dampening = dampening
        self.decoupled = decoupled
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.decoupled = True

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.beta1_prod = []
        self.step = 0
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
            self.beta1_prod.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="beta1_prod", initializer="ones"
                )
            )
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        param_size = 0
        exp_avg_sq_hat_sum = 0.
        self.step += 1
        
        for p, g in zip(trainable_variables, grads):
            lr = tf.cast(learning_rate, p.dtype)
            
            param_size += tf.size(p)
            
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            
            bias_correction2 = 1 - self.beta2 ** self.step

            if self.weight_decay != 0 and self.decoupled == False:
                grads[self._get_variable_index(p)] = g + p * self.weight_decay
            elif self.weight_decay != 0 and self.decoupled == True:
                p.assign(p * (1 - lr * self.weight_decay))
                
            exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * tf.square(g))
            
            exp_avg_sq_hat_sum += tf.reduce_sum(exp_avg_sq) / bias_correction2
        
        # Calculate the mean of all elements in exp_avg_sq_hat
        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / tf.cast(param_size, exp_avg_sq_hat_sum.dtype)
        
        for p, g in zip(trainable_variables, grads):
            lr = tf.cast(learning_rate, p.dtype)
            
            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            beta1_prod = self.beta1_prod[self._get_variable_index(p)]
            
            bias_correction2 = 1 - self.beta2 ** self.step

            exp_avg_sq_hat = exp_avg_sq / bias_correction2
            beta1 = tf.clip_by_value(1. - tf.pow(exp_avg_sq_hat / exp_avg_sq_hat_mean, 1.0 / (3 - 2 * self.dampening)) * self.beta0, 
                             clip_value_min=0., clip_value_max=1 - self.epsilon)
            beta3 = tf.pow((1. - beta1), self.dampening)

            
            beta1_prod.assign(beta1_prod * beta1)
            bias_correction1 = 1 - beta1_prod
            
            exp_avg.assign(exp_avg * beta1 + beta3 * g)
            exp_avg_hat = exp_avg / bias_correction1 * math.pow(self.beta0, 1. - self.dampening)
            
            p.assign_add(exp_avg_hat * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta0": self.beta0,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "dampening": self.dampening,
                "decoupled": self.decoupled,
                "step": self.iterations.numpy(),
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass