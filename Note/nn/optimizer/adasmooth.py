""" AdaSmooth
https://arxiv.org/abs/2204.00825

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AdaSmooth(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.5,
        beta2=0.99,
        epsilon=1e-6,
        weight_decay=0.0,
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
        name="adasmooth",
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
        for var in self._trainable_variables:
            self.prev_param[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="prev_param"
                                                    )
            
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="s"
                                                    )
            
            self.n[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="n"
                                                    )
            
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.prev_param = []
        self.s = []
        self.n = []
        self.exp_avg_sq = []
        for var in var_list:
            self.prev_param.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="prev_param"
                )
            )
            self.s.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="s"
                )
            )
            self.n.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="n"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdaSmooth does not support sparse gradients')
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient.assign_add(variable * self.weight_decay)
        
        prev_param = self.prev_param[self._get_variable_index(variable)]
        p_diff = variable - prev_param
        
        s = self.s[self._get_variable_index(variable)]
        n = self.n[self._get_variable_index(variable)]
        s.assign_add(p_diff)
        n.assign_add_(tf.abs(p_diff))
      
        c = tf.abs(tf.reduce_sum(s)) / tf.reduce_sum(n)  # e_t
        c = c * (self.beta2 - self.beta1) + (1.0 - self.beta2)
        
        c_p2 = tf.pow(c, 2)
        
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg_sq.assign(exp_avg_sq * (1.0 - c_p2) + c_p2 * (gradient * gradient))
            
        step_size = tf.fill(tf.shape(exp_avg_sq), lr)
        step_size = step_size / tf.sqrt(exp_avg_sq + self.epsilon) * gradient

        variable.assign_add(-step_size)

        self.prev_param[self._get_variable_index(variable)] = tf.Variable(variable)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass