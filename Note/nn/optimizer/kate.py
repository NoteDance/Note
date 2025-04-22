""" Kate
https://arxiv.org/abs/2403.02648

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class Kate(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        epsilon=1e-8,
        weight_decay=0.0,
        delta=0.0,
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
        name="kate",
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
        self.epsilon = epsilon
        self.delta = delta
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.m[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="m"
                                                    )
            self.b[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="b"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.m = []
        self.b = []
        for var in var_list:
            self.m.append(self.add_variable_from_reference(
                                reference_variable=var, name="m"
                                                    ))
            self.b.append(self.add_variable_from_reference(
                                reference_variable=var, name="b"
                                                    ))

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Kate does not support sparse gradients')
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay

        grad_p2 = gradient * gradient

        m = self.m[self._get_variable_index(variable)]
        b = self.b[self._get_variable_index(variable)]
        b.assign(b * b + grad_p2 + self.epsilon)
        
        m.assign(tf.sqrt(m * m + grad_p2 * self.delta + grad_p2 / b))

        update = m * gradient / b

        variable.assign_add(update * -lr)

        b.assign(tf.sqrt(b))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass