import tensorflow as tf
from keras.src.optimizers import optimizer


class SignSGD(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=0.0,
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
        name="signsgd",
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
        self.weight_decouple = weight_decouple
        self.maximize = maximize
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum_buffer[self._get_variable_index(var)].assign(tf.zeros_like(var))

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        for var in var_list:
            self.momentum_buffer.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                ))

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'SignSGD does not support sparse gradient.')
            
        lr = tf.cast(learning_rate, variable.dtype)
        
        if self.maximize:
            gradient = -gradient
            
        if self.momentum > 0.0:
            buf = self.momentum_buffer[self._get_variable_index(variable)]
            buf.assign(buf * self.momentum + gradient * (1.0 - self.momentum))
        else:
            buf = gradient

        variable.assign_add((tf.sign(buf) if not buf.dtype.is_complex else tf.sign(buf)) * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "maximize": self.maximize,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass