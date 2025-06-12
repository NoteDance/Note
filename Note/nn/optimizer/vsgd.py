""" VSGD
https://arxiv.org/abs/2404.06549

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class VSGD(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-1,
        epsilon=1e-8,
        weight_decay=0.0,
        weight_decouple=True,
        ghattg=30.0,
        ps=1e-8,
        tau1=0.81,
        tau2=0.9,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="vsgd",
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
        self.weight_decouple = weight_decouple
        self.ghattg = ghattg
        self.ps = ps
        self.tau1 = tau1
        self.tau2 = tau2
        self.maximize = maximize
        
        self.pa2 = 2.0 * ps + 1.0 + 1e-4
        self.pbg2 = 2.0 * ps
        self.pbhg2 = 2.0 * ghattg * ps
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.mug[self._get_variable_index(var)].assign(tf.zeros_like(var))
            self.bg[self._get_variable_index(var)].assign(tf.zeros_like(var))
            self.bhg[self._get_variable_index(var)].assign(tf.zeros_like(var))

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.mug = []
        self.bg = []
        self.bhg = []
        for var in var_list:
            self.mug.append(self.add_variable_from_reference(
                    reference_variable=var, name="mug"
                ))
            self.bg.append(self.add_variable_from_reference(
                    reference_variable=var, name="bg"
                ))
            self.bhg.append(self.add_variable_from_reference(
                    reference_variable=var, name="bhg"
                ))

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'VSGD does not support sparse gradient.')
        
        step = tf.cast(self.iterations + 1, variable.dtype)
            
        lr = tf.cast(learning_rate, variable.dtype)
        
        rho1 = tf.pow(step, -self.tau1)
        rho2 = tf.pow(step, -self.tau2)
        
        if self.maximize:
            gradient = -gradient
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * lr))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        bg = self.bg[self._get_variable_index(variable)]
        bhg = self.bhg[self._get_variable_index(variable)]
        
        def true_fn():
            sg = self.pbg2 / (self.pa2 - 1.0)
            shg = self.pbhg2 / (self.pa2 - 1.0)
            return sg, shg
        def false_fn():
            sg = bg / self.pa2
            shg = bhg / self.pa2
            return sg, shg
        sg, shg = tf.cond(step == 1, true_fn, false_fn)
        
        mug = self.mug[self._get_variable_index(variable)]
        mug_prev = tf.identity(mug)
        
        mug.assign((mug * shg + gradient * sg) / (sg + shg))
        
        sigg = (sg * shg) / (sg + shg)
        mug_sq = tf.pow(mug, 2) + sigg
        
        bg2 = self.pbg2 + mug_sq - 2.0 * mug * mug_prev + tf.pow(mug_prev, 2)
        bhg2 = self.pbhg2 + mug_sq - 2.0 * gradient * mug + tf.pow(gradient, 2)
        
        bg.assign(bg * (1.0 - rho1) + bg2 * rho1)
        bhg.assign(bhg * (1.0 - rho2) + bhg2 * rho2)
            
        variable.assign_add(-1.0 * lr / (tf.sqrt(mug_sq) + self.epsilon) * mug)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "ghattg": self.ghattg,
                "ps": self.ps,
                "tau1": self.tau1,
                "tau2": self.tau2,
                "maximize": self.maximize,
                "pa2": self.pa2,
                "pbg2": self.pbg2,
                "pbhg2": self.pbhg2,       
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass