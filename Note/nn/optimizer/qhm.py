""" QHM
Implements quasi-hyperbolic momentum (QHM)  optimization algorithm.
It has been proposed in `Quasi-hyperbolic momentum and Adam for deep
learning`__.
https://arxiv.org/abs/1810.06801

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class QHM(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=0.0,
        momentum=0.0,
        nu=0.7,
        weight_decay_type="grad",
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="qhm",
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
        self.momentum = momentum
        self.nu = nu
        self.weight_decay_type = weight_decay_type
        self.GRAD = "grad"
        self.DIRECT = "direct"

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        for var in var_list:
            self.momentum_buffer.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        nu1, nu2 = self.Nus2
        
        d_p = gradient

        if self.weight_decay != 0:
            if self.weight_decay_type == self.GRAD:
                d_p += variable * self.weight_decay
            else:
                variable.assign(variable * (1.0 - lr * self.weight_decay))

        momentum_buffer = self.momentum_buffer[self._get_variable_index(variable)]
        momentum_buffer.assign(momentum_buffer * self.momentum + d_p * (1.0 - self.momentum))

        variable.assign_add(momentum_buffer * -lr * self.nu)
        variable.assign_add(d_p * -lr * (1.0 - self.nu))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "nu": self.nu,
                "weight_decay_type": self. weight_decay_type,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass