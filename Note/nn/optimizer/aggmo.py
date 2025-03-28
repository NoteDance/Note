""" AggMo
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AggMo(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.0, 0.9, 0.99),
        weight_decay=0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="aggmo",
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
    
    @classmethod
    def from_exp_form(
        cls,
        lr = 1e-3,
        a = 0.1,
        k = 3,
        weight_decay: float = 0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid parameter k: {}".format(k))

        betas = [1 - a**i for i in range(k)]  # type: List[float]
        return cls(lr, betas, weight_decay)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        for i,var in enumerate(var_list):
            self.momentum_buffer.append(dict())
            for beta in self.betas:
                self.momentum_buffer[i][beta]=self.add_variable_from_reference(
                                            reference_variable=var, name="momentum_buffer"
                                        )

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        total_mom = float(len(self.betas))
        
        d_p = tf.Variable(gradient)
        if self.weight_decay != 0:
            d_p.assign_add(variable * self.weight_decay)
        for beta in self.betas:
            buf = self.momentum_buffer[self._get_variable_index(variable)][beta]
            buf.assign(buf * beta + d_p)
            variable.assign_sub(buf * (lr / total_mom))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass