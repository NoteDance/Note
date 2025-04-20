""" MSVAG
https://arxiv.org/abs/1705.07774

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class MSVAG(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        beta=0.9,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="msvag",
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
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="s"
                                                    )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.s = []
        self.step = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.s.append(self.add_variable_from_reference(
                                reference_variable=var, name="s"
                                                    ))
            self.step.append(0)
    
    @staticmethod
    def get_rho(beta_power: float, beta: float) -> float:
        r"""Get rho."""
        rho: float = (1.0 - beta_power ** 2) * (1.0 - beta) ** 2  # fmt: skip
        rho /= (1.0 - beta) * (1.0 - beta_power) ** 2
        return min(rho, 0.9999)
    
    def nan_to_num(tensor, nan=0.0, out=None):
        result = tf.where(tf.math.is_nan(tensor), tf.constant(nan, dtype=tensor.dtype), tensor)
        if out is not None:
            out.assign(result)
            return out
        return result

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        beta_power = self.beta ** self.step[self._get_variable_index(variable)]
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'MSVAG does not support sparse gradients')

        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta + gradient * (1.0 - self.beta))
        exp_avg_sq.assign(exp_avg_sq * self.beta + gradient * gradient * (1.0 - self.beta))
        
        m = exp_avg / beta_power
        v = exp_avg_sq / beta_power

        rho = self.get_rho(beta_power, self.beta)

        m_p2 = tf.pow(m, 2)
        s = (v - m_p2) / (1.0 - rho)

        factor = m_p2 / (m_p2 + rho * s)
        factor = self.nan_to_num(factor, nan=0.0)
        factor = tf.clip_by_value(factor, 0.0, 1.0)

        variable.assign_add(m * factor * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass