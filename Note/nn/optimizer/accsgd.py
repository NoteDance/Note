""" AccSGD
Implements the algorithm proposed in https://arxiv.org/pdf/1704.08227.pdf, which is a provably accelerated method 
for stochastic optimization. This has been employed in https://openreview.net/forum?id=rJTutzbA- for training several 
deep learning models of practical interest. This code has been implemented by building on the construction of the SGD 
optimization module found in pytorch codebase.

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AccSGD(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=0,
        kappa=1000.0,
        xi=10.0,
        smallConst=0.7,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="accsgd",
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
        self.kappa = kappa
        self.xi = xi
        self.smallConst = smallConst

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        for var in var_list:
            self.momentum_buffer.append(tf.Variable(var))

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        large_lr = (lr*self.kappa)/(self.smallConst)
        Alpha = 1.0 - ((self.smallConst*self.smallConst*self.xi)/self.kappa)
        Beta = 1.0 - Alpha
        zeta = self.smallConst/(self.smallConst+Beta)
        d_p = gradient
        if self.weight_decay != 0:
            d_p.assign_add_(self.weight_decay, variable)
        buf = self.momentum_buffer[self._get_variable_index(variable)]
        buf.assign(buf * (1.0/Beta)-1.0)
        buf.assign_add(-large_lr * d_p)
        buf.assign_add(variable)
        buf.assign(buf * Beta)

        variable.assign_add(-lr * d_p)
        variable.assign(variable * zeta)
        variable.assign_add((1.0-zeta) * buf)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kappa": self.kappa,
                "xi": self.xi,
                "smallConst": self.smallConst,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass