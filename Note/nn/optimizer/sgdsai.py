""" SGDSaI
https://arxiv.org/abs/2412.11768

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class SGDSaI(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        epsilon=1e-8,
        weight_decay=1e-2,
        momentum=0.9,
        weight_decouple=True,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sgdsai",
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
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.maximize = maximize
        self.has_warmup = False
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum_buffer[self._get_variable_index(var)].assign(tf.zeros_like(var))
            self.gsnr.assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.gsnr = []
        for var in var_list:
            self.momentum_buffer.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                ))
            self.gsnr.append(tf.Variable(tf.zeros((), dtype=var.dtype)))
            self._track_variable(self.gsnr[-1])
    
    def warmup_step(self, params, grads):
        for p, grad in zip(params, grads):
            if self.maximize:
                grad = -grad

            sigma = tf.where(tf.rank(grad) > 1 and tf.shape(grad)[0] != 1, tf.math.reduce_std(tf.where(tf.math.is_nan(grad), 0.0, grad)), 0.0)
            grad_norm = tf.norm(grad)
            
            def true_fn():
                return grad_norm / (sigma + self.epsilon)
            def false_fn():
                return grad_norm
            g_snr = tf.cond(sigma != 0.0, true_fn, false_fn)

            self.gsnr[self._get_variable_index(p)].assign(g_snr)

        self.has_warmup = True

        return

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if not self.has_warmup:
            self.warmup_step(trainable_variables, grads)
            
        for variable, gradient in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(gradient):
                raise RuntimeError(
                    'SGDSaI does not support sparse gradient.')
            
            lr = tf.cast(learning_rate, variable.dtype)
            
            if self.maximize:
                gradient = -gradient
                
            if self.momentum > 0.0:
                buf = self.momentum_buffer[self._get_variable_index(variable)]
                buf.assign(buf * self.momentum + gradient * (1.0 - self.momentum))
            else:
                buf = gradient
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * lr))
            elif self.weight_decay > 0.0:
                gradient += variable * self.weight_decay
            
            variable.assign_add(buf * -lr * self.gsnr[self._get_variable_index(variable)])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "maximize": self.maximize,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass