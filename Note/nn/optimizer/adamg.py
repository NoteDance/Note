""" AdamG
https://arxiv.org/abs/2405.04376

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AdamG(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        betas=(0.95, 0.999, 0.95),
        epsilon=1e-8,
        weight_decay=0.0,
        p=0.2,
        q=0.24,
        weight_decouple=False,
        fixed_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamg",
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
        self.betas = betas
        self.epsilon = epsilon
        self.p = p
        self.q = q
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.m[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="m"
                                                    )
            self.v[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="v"
                                                    )
            self.r[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="r"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.m = []
        self.v = []
        self.r = []
        if self.variant == 'exp':
            self.v_kk = []
        for var in var_list:
            self.m.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="m"
                )
            )
            self.v.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="v"
                )
            )
            self.r.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="r"
                )
            )
    
    def s(self, p):
        r"""Numerator function f(x) = p * x^q."""
        return tf.pow(p, self.q) * self.p

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdamG does not support sparse gradients, please consider SparseAdam instead')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        beta1, beta2, beta3 = self.betas
        
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = tf.minimum(lr, 1.0 / tf.sqrt(step))
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        m = self.m[self._get_variable_index(variable)]
        v = self.v[self._get_variable_index(variable)]
        r = self.r[self._get_variable_index(variable)]
        v.assign(v * beta2 + gradient * gradient * (1.0 - beta2))
        r.assign(r * beta3 + self.s(v) * (1.0 - beta3))
        m.assign(m * beta1 + r * gradient * (1.0 - beta1))
        
        update = (m / bias_correction1) / (tf.sqrt(v / bias_correction2) + self.epsilon)

        variable.assign_add(update * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "p": self.p,
                "q": self.q,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass