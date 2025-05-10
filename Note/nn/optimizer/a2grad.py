""" A2Grad
https://arxiv.org/abs/1810.00553

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class A2Grad(optimizer.Optimizer):
    def __init__(
        self,
        beta=10.0,
        lips=10.0,
        rho=0.5,
        variant='uni',
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="a2grad",
        **kwargs,
    ):
        super().__init__(
            learning_rate=1.,
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
        self.lips = lips
        self.rho = rho
        self.variant = variant
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.v_k[self._get_variable_index(var)].assign(0)
            self.alpha_k[self._get_variable_index(var)].assign(1)
            self.x_k[self._get_variable_index(var)].assign(var)
            if self.variant == 'exp':
                self.v_kk[self._get_variable_index(var)].assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.v_k = []
        self.alpha_k = []
        self.avg_grad = []
        self.x_k = []
        if self.variant == 'exp':
            self.v_kk = []
        for var in var_list:
            self.v_k.append(tf.Variable(tf.zeros((), var.dtype)))
            self._track_variable(self.v_k[-1])
            self.alpha_k.append(tf.Variable(tf.ones((), var.dtype)))
            self._track_variable(self.alpha_k[-1])
            self.avg_grad.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="avg_grad"
                )
            )
            self.x_k.append(tf.Variable(var))
            if self.variant == 'exp':
                self.v_kk.append(tf.Variable(tf.zeros((), var.dtype)))
                self._track_variable(self.v_kk[-1])

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'A2Grad does not support sparse gradients, please consider SparseAdam instead')
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        gamma_k = 2.0 * self.lips / (step + 1)
        alpha_k_1 = 2.0 / (step + 3)
        
        avg_grad = self.avg_grad[self._get_variable_index(variable)]
        def true_fn():
            avg_grad.assign(gradient)
        def false_fn():
            pass
        tf.cond(step == 1, true_fn, false_fn)
        
        avg_grad.assign_add((gradient - avg_grad) * step + 1)
        
        delta_k = gradient
        delta_k += avg_grad * -1.0
        
        delta_k_sq = tf.reduce_sum(tf.pow(delta_k, 2))
        
        v_k = self.v_k[self._get_variable_index(variable)]
        if self.variant in ('uni', 'inc'):
            if self.variant == 'inc':
                v_k.assign(v_k * (step / (step + 1)) ** 2)
            v_k.assign_add(delta_k_sq)
        else:
            v_kk = self.v_kk[self._get_variable_index(variable)]
            v_kk.assign(v_kk * self.rho + delta_k_sq * (1.0 - self.rho))
            v_k.assign(tf.maximum(v_kk, v_k))
        
        h_k = tf.sqrt(v_k)
        if self.variant != 'uni':
            h_k *= tf.sqrt(step + 1)
        
        coefficient = -1.0 / (gamma_k + self.beta * h_k)
        
        x_k = self.x_k[self._get_variable_index(variable)]
        x_k.assign_add(gradient * coefficient)
        
        variable.assign(variable * (1.0 - alpha_k_1) + x_k * alpha_k_1)
        variable.assign_add(gradient * (1.0 - alpha_k_1) * self.alpha_k * coefficient)

        self.alpha_k[self._get_variable_index(variable)].assign(alpha_k_1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "lips": self.lips,
                "rho": self.rho,
                "variant": self.variant,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass