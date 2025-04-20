""" PNM
https://arxiv.org/abs/2103.17182

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class PNM(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=1.0,
        epsilon=1e-8,
        weight_decay=0.0,
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
        name="pnm",
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
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
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
            self.pos_momentum[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="pos_momentum"
                                                    )
            self.neg_momentum[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="neg_momentum"
                                                    )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.pos_momentum = []
        self.neg_momentum = []
        self.step = []
        for var in var_list:
            self.pos_momentum.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="pos_momentum"
                )
            )
            self.neg_momentum.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="neg_momentum"
                )
            )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)  # fmt: skip
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'PNM does not support sparse gradients')
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        if self.step[self._get_variable_index(variable)] % 2 == 1:
            pos_momentum = self.pos_momentum[self._get_variable_index(variable)]
            neg_momentum = self.neg_momentum[self._get_variable_index(variable)]
        else:
            neg_momentum = self.pos_momentum[self._get_variable_index(variable)]
            pos_momentum = self.neg_momentum[self._get_variable_index(variable)]
        
        pos_momentum.assign(pos_momentum * self.beta1 ** 2 + gradient * (1.0 - self.beta1 ** 2))  # fmt: skip

        delta_p = pos_momentum * (1 + self.beta2) + neg_momentum * -self.beta2 * (1.0 / noise_norm)
        variable.assign_add(delta_p * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass