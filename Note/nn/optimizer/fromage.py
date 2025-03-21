""" Fromage
https://arxiv.org/abs/2002.03432

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class Fromage(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        p_bound=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="fromage",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=None,
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
        self.p_bound = p_bound
    
    def reset(self):
        for var in self._trainable_variables:
            if self.p_bound is not None:
                self.max[self._get_variable_index(var)] =  tf.norm(var) * self.p_bound

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.max = []
        for var in var_list:
            self.max.append(tf.norm(var) * self.p_bound)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
                
        pre_factor = math.sqrt(1 + lr ** 2)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Fromage does not support sparse gradients')
        
        p_norm = tf.norm(variable) 
        g_norm = tf.norm(gradient)
        
        if tf.get_static_value(p_norm) > 0.0 and tf.get_static_value(g_norm) > 0.0:
            variable.assign_add(gradient * (p_norm / g_norm) * -lr)
        else:
            variable.assign_add(gradient * -lr)
        
        variable.assign(variable / pre_factor)
        
        if self.p_bound is not None:
            p_norm = tf.norm(variable) 
            if tf.get_static_value(p_norm) > self.max[self._get_variable_index(variable)]:
                variable.assign(variable * self.max[self._get_variable_index(variable)] / p_norm)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p_bound": self.p_bound,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass