""" AdaMax
https://arxiv.org/abs/1910.12249

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AdaMax(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.0,
        r=0.95,
        adanorm=False,
        adam_debias=False,
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
        name="adamax",
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
        self.r = r
        self.adanorm = adanorm
        self.adam_debias = adam_debias
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_inf[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_inf"
                                                    )
            if self.adanorm:
                self.exp_grad_norm[self._get_variable_index(var)].assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_inf = []
        if self.adanorm:
            self.exp_grad_norm = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_inf.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_inf"
                )
            )
            if self.adanorm:
                self.exp_grad_norm.append(tf.Variable(tf.zeros((), var.dtype)))
                self._track_variable(self.exp_grad_norm[-1])

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdaMax does not support sparse gradients, please consider SparseAdam instead')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        beta1, beta2 = self.betas
        
        bias_correction1 = 1 - beta1 ** step
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        if not self.adanorm:
            s_grad = gradient
        else:
            grad_norm = tf.linalg.norm(gradient)
            exp_grad_norm = self.exp_grad_norm[self._get_variable_index(variable)]
            exp_grad_norm.assign(exp_grad_norm * self.r + grad_norm * (1.0 - self.r))
            def true_fn():
                return gradient * exp_grad_norm / grad_norm
            def false_fn():
                return gradient
            s_grad = tf.cond(exp_grad_norm > grad_norm, true_fn, false_fn)
            
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_inf = self.exp_inf[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * beta1 + s_grad * (1.0 - beta1))
        
        norm_buf = tf.concat([
            tf.expand_dims(exp_inf.assign(exp_inf * beta2), 0),
            tf.expand_dims(tf.abs(gradient) + self.epsilon, 0)
        ], axis=0)
        exp_inf.assign(tf.reduce_max(norm_buf, axis=0))
        
        step_size = lr if self.adam_debias else lr / bias_correction1

        variable.assign_add(-step_size * exp_avg / exp_inf)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "r": self.r,
                "adanorm": self.adanorm,
                "adam_debias": self.adam_debias,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass