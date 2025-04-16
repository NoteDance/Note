""" DAdaptSGD
https://arxiv.org/abs/2301.07733

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class DAdaptSGD(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        weight_decay=0.0,
        momentum=0.9,
        d0=1e-6,
        growth_rate=float('inf'),
        weight_decouple=True,
        fixed_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="dadaptsgd",
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
        self.lr = learning_rate
        self.momentum = momentum
        self.d0_ = d0
        self.growth_rate = growth_rate
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self.sk_sq = tf.Variable(0.0)
        self.numerator_weighted = tf.Variable(0.0)
        self.global_grad_norm = tf.Variable(tf.zeros(1, dtype=tf.float32))
        self.g0_norm = tf.Variable(0.0)
        self.d0 = tf.Variable(self.d0_)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.global_grad_norm)
        self._track_variable(self.g0_norm)
        self._track_variable(self.d0)
        self.step = 0
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
            self.z[self._get_variable_index(var)] =  tf.Variable(var)
            self._track_variable(self.z[self._get_variable_index(var)])
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="s"
                                                    )
            self.x0[self._get_variable_index(var)] =  tf.Variable(var)
            self._track_variable(self.x0[self._get_variable_index(var)])
            
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.z = []
        self.s = []
        self.x0 = []
        self.sk_sq = tf.Variable(0.0)
        self.numerator_weighted = tf.Variable(0.0)
        self.global_grad_norm = tf.Variable(tf.zeros((), dtype=tf.float32))
        self.g0_norm = tf.Variable(0.0)
        self.d0 = tf.Variable(self.d0_)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.global_grad_norm)
        self._track_variable(self.g0_norm)
        self._track_variable(self.d0)
        self.step = 0
        for var in var_list:
            self.z.append(tf.Variable(var))
            self._track_variable(self.z[-1])
            self.s.append(self.add_variable_from_reference(
                                reference_variable=var, name="s"
                                                    ))
            self.x0.append(tf.Variable(var))
            self._track_variable(self.x0[-1])
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.step == 0:
            for grad in grads:
                self.global_grad_norm.assign_add(tf.cast(tf.pow(tf.norm(grad), 2), tf.float32))
            self.g0_norm.assign(tf.sqrt(self.global_grad_norm))
        
        def update_fn():
            d = self.d0
            d_lr = d * self.lr / self.g0_norm
            
            for variable, grad in zip(trainable_variables, grads):
                if tf.keras.backend.is_sparse(grad):
                    raise RuntimeError(
                        'DAdaptSGD does not support sparse gradients')
                
                if self.weight_decouple:
                    variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else self.lr)))
                
                d_lr = tf.cast(d_lr, variable.dtype)
                
                s = self.s[self._get_variable_index(variable)]
                self.numerator_weighted.assign_add(tf.cast(tf.tensordot(tf.reshape(grad, [-1]), tf.reshape(s, [-1]), axes=1) * d_lr, tf.float32))
                
                s.assign_add(grad * d_lr)
                self.sk_sq.assign_add(tf.cast(tf.reduce_sum(tf.pow(s, 2)), tf.float32))
            
            if self.lr > 0.0:
                d_hat = 2.0 * self.numerator_weighted / tf.sqrt(self.sk_sq)
                d = tf.maximum(self.d0, tf.minimum(d_hat, self.d0 * self.growth_rate))
            
            for variable, grad in zip(trainable_variables, grads):
                z = self.z[self._get_variable_index(variable)]
                z.assign(self.x0[self._get_variable_index(variable)] - self.s[self._get_variable_index(variable)])
                
                variable.assign(variable * self.momentum + z * (1.0 - self.momentum))
            
        def no_update_fn():
            pass
        
        tf.cond(self.g0_norm == 0, no_update_fn, update_fn)
        
        self.step += 1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "momentum": self.momentum,
                "d0_": self.d0_,
                "growth_rate": self.growth_rate,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "step": self.iterations.numpy(),
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass