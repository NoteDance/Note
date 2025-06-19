""" AdamC
https://arxiv.org/abs/2506.02285

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AdamC(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        ams_bound=False,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamc",
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
        self.ams_bound = ams_bound
        self.maximize = maximize
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)].assign(tf.zeros_like(var))
            self.exp_avg_sq[self._get_variable_index(var)].assign(tf.zeros_like(var))
            if self.ams_bound:
                self.max_exp_avg_sq[self._get_variable_index(var)].assign(tf.zeros_like(var))

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        if self.ams_bound:
            self.max_exp_avg_sq = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                ))
            if self.ams_bound:
                self.max_exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_sq"
                    ))

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdamC does not support sparse gradient.')
        
        if variable.dtype.is_complex:
            raise RuntimeError(
                'AdamC does not support complex parameter.')
        
        step = tf.cast(self.iterations + 1, variable.dtype)
            
        lr = tf.cast(learning_rate, variable.dtype)
        
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
        
        if self.maximize:
            gradient = -gradient
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        if self.ams_bound:
            max_exp_avg_sq = self.max_exp_avg_sq[self._get_variable_index(variable)]
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + gradient * gradient * (1.0 - self.beta2))
        
        if self.ams_bound:
            max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq))
            de_nom = max_exp_avg_sq + self.epsilon
        else:
            de_nom = exp_avg_sq + self.epsilon
        de_nom = tf.sqrt(de_nom) + self.epsilon
        de_nom /= bias_correction2_sq
        
        variable.assign_add(-lr * (exp_avg / bias_correction1) / de_nom)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "ams_bound": self.ams_bound,
                "maximize": self.maximize,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass