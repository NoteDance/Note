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
        self.lr = learning_rate
        self.p_bound = p_bound
    
    def reset(self):
        iterations = tf.Variable(
                0,
                name="iteration",
                dtype=tf.int64,
                trainable=False,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
        self._track_variable(iterations)
        self._iterations = iterations
        for var in self._trainable_variables:
            if self.p_bound is not None:
                self.max[self._get_variable_index(var)] =  tf.Variable(tf.norm(var) * self.p_bound)
                self._track_variable(self.max[self._get_variable_index(var)])

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.max = []
        for var in var_list:
            if self.p_bound is not None:
                self.max.append(tf.Variable(tf.norm(var) * self.p_bound))
                self._track_variable(self.max[-1])

    def update_step(self, gradient, variable, learning_rate):
        pre_factor = math.sqrt(1 + self.lr ** 2)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Fromage does not support sparse gradients')
        
        p_norm = tf.norm(variable) 
        g_norm = tf.norm(gradient)
        
        combined_condition = tf.logical_and(tf.greater(p_norm, 0.0), tf.greater(g_norm, 0.0))

        def true_fn():
            return variable + gradient * (p_norm / g_norm) * -self.lr
        def false_fn():
            return variable + gradient * -self.lr
        
        variable.assign(tf.cond(combined_condition, true_fn, false_fn))
        
        variable.assign(variable / pre_factor)
        
        if self.p_bound is not None:
            p_norm = tf.norm(variable)
            def true_fn():
                variable.assign(variable * self.max[self._get_variable_index(variable)] / p_norm)
            def false_fn():
                pass
            tf.cond(p_norm > self.max[self._get_variable_index(variable)], true_fn, false_fn)

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