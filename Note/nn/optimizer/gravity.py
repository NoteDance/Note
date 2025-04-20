""" Gravity
https://arxiv.org/abs/2101.09192

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class Gravity(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        alpha=0.01,
        beta=0.9,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="gravity",
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
        self.alpha = alpha
        self.beta = beta
    
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
            self.v[self._get_variable_index(var)] =  tf.Variable(tf.random.normal(shape=var.shape,
                                                       mean=0.0,
                                                       stddev=self.alpha / self._learning_rate,
                                                       dtype=var.dtype))
            self._track_variable(self.v[self._get_variable_index(var)])
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.v = []
        self.step = []
        for var in var_list:
            self.v.append(tf.Variable(tf.random.normal(shape=var.shape,
                                                       mean=0.0,
                                                       stddev=self.alpha / self._learning_rate,
                                                       dtype=var.dtype)))
            self._track_variable(self.v[-1])
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
                
        self.step[self._get_variable_index(variable)] += 1
        
        beta_t = (self.beta * self.step[self._get_variable_index(variable)] + 1) / (self.step[self._get_variable_index(variable)] + 2)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Gravity does not support sparse gradients')
        
        v = self.v[self._get_variable_index(variable)]

        m = 1.0 / tf.reduce_max(tf.abs(gradient))
        zeta = gradient / (1.0 + (gradient / m) ** 2)

        v.assign(v * beta_t + zeta * (1.0 - beta_t))

        variable.assign_add(v * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass