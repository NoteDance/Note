""" QHAdam
Implements the QHAdam optimization algorithm.
It has been proposed in `Adaptive methods for Nonconvex Optimization`__.
https://arxiv.org/abs/1810.06801

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class QHAdam(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        Nus2=(1.0, 1.0),
        decouple_weight_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="qhadam",
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
        self.Nus2 = Nus2
        self.decouple_weight_decay = decouple_weight_decay
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.nesterov = False

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.beta1_weight = 0.0
        self.beta2_weight = 0.0
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.grad_buffer.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        nu1, nu2 = self.Nus2
        
        d_p = gradient
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                "QHAdam does not support sparse gradients, "
                "please consider SparseAdam instead"
            )

        if self.weight_decay != 0:
            if self.decouple_weight_decay:
                variable.assign(variable * (1 - lr * self.weight_decay))
            else:
                d_p.assign_add(variable * self.weight_decay)

        d_p_sq = d_p * d_p

        self.beta1_weight = 1.0 + self.beta1 * self.beta1_weight
        self.beta2_weight = 1.0 + self.beta2 * self.beta2_weight

        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]

        beta1_adj = 1.0 - (1.0 / self.beta1_weight)
        beta2_adj = 1.0 - (1.0 / self.beta2_weight)
        exp_avg.assign(exp_avg * beta1_adj + d_p * (1.0 - beta1_adj))
        exp_avg_sq.assign(exp_avg_sq * beta2_adj + d_p_sq * (1.0 - beta2_adj))

        avg_grad = exp_avg * nu1
        if nu1 != 1.0:
            avg_grad += d_p * (1.0 - nu1)

        avg_grad_rms = exp_avg_sq * nu2
        if nu2 != 1.0:
            avg_grad_rms += d_p_sq * (1.0 - nu2)
        avg_grad_rms = tf.sqrt(avg_grad_rms)
        if self.epsilon != 0.0:
            avg_grad_rms += self.epsilon

        variable.assign_add(-lr * avg_grad / avg_grad_rms)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "Nus2": self.Nus2,
                "decouple_weight_decay": self.decouple_weight_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass