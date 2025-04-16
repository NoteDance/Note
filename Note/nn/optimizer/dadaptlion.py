""" DAdaptLion
https://arxiv.org/abs/2301.07733

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class DAdaptLion(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        d0=1e-6,
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
        name="dadaptlion",
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.d0_ = d0
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self.numerator_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.numerator_accumulator = tf.Variable(0.0)
        self.d0 = tf.Variable(self.d0_)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.numerator_accumulator)
        self._track_variable(self.d0)
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
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="s"
                                                    )
            
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.s = []
        self.numerator_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.numerator_accumulator = tf.Variable(0.0)
        self.d0 = tf.Variable(self.d0_)
        self._track_variable(self.numerator_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.numerator_accumulator)
        self._track_variable(self.d0)
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.s.append(self.add_variable_from_reference(
                                reference_variable=var, name="s"
                                                    ))
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        beta2_sq = math.sqrt(self.beta2)
        
        d_lr = self.d0 * self.lr
        
        for variable, grad in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(grad):
                raise RuntimeError(
                    'DAdaptLion does not support sparse gradients')
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else self.lr)))
            elif self.weight_decay > 0.0:
                grad += variable * self.weight_decay
            
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            s = self.s[self._get_variable_index(variable)]
            
            d_lr = tf.cast(d_lr, variable.dtype)
            
            update = tf.math.sign(exp_avg * self.beta1 + grad * (1.0 - self.beta1))
            variable.assign_add(update * -d_lr)
            
            exp_avg.assign(exp_avg * self.beta2 + grad * (1.0 - self.beta2) * d_lr)
            
            self.numerator_accumulator.assign_add(tf.cast(tf.tensordot(tf.reshape(update, [-1]), tf.reshape(s, [-1]), axes=1) * d_lr, tf.float32))
            s.assign(s * beta2_sq + update * (1.0 - beta2_sq) * d_lr)
            
            self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
        
        self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_accumulator * (1.0 - beta2_sq))
        
        def update_fn():
            d = self.d0
            if self.lr > 0.0:
                d_hat = self.numerator_weighted / ((1.0 - beta2_sq) * self.sk_l1)
                d = tf.maximum(self.d0, d_hat)
            
            self.d0.assign(d)
            
        def no_update_fn():
            pass
        
        tf.cond(self.sk_l1 == 0, no_update_fn, update_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "d0_": self.d0_,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass