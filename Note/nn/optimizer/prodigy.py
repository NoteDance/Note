""" Prodigy
https://arxiv.org/abs/2306.06101

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class Prodigy(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        beta3=None,
        epsilon=1e-8,
        weight_decay=0.0,
        d0=1e-6,
        d_coef=1.0,
        growth_rate=float('inf'),
        weight_decouple=True,
        fixed_decay=False,
        bias_correction=False,
        safeguard_warmup=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="prodigy",
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
        self.beta3 = beta3
        self.epsilon = epsilon
        self.d_ = d0
        self.d0_ = d0
        self.d_max_ = d0
        self.d_coef = d_coef
        self.growth_rate = growth_rate
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.bias_correction = bias_correction
        self.safeguard_warmup = safeguard_warmup
    
    def reset(self):
        self.d = tf.Variable(self.d_)
        self.d0 = tf.Variable(self.d0_)
        self.d_max = tf.Variable(self.d_max_)
        self.d_hat = tf.Variable(self.d_)
        self.d_de_nom = tf.Variable(0.0)
        self.d_numerator = tf.Variable(0.0)
        self._track_variable(self.d)
        self._track_variable(self.d0)
        self._track_variable(self.d_max)
        self._track_variable(self.d_hat)
        self._track_variable(self.d_de_nom)
        self._track_variable(self.d_numerator)
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
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="s"
                                                    )
            self.p0 = tf.Variable(var)
            self._track_variable(self.p0[self._get_variable_index(var)])
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.s = []
        self.p0 = []
        self.exp_avg = []
        self.exp_avg_sq = []
        self.d = tf.Variable(self.d_)
        self.d0 = tf.Variable(self.d0_)
        self.d_max = tf.Variable(self.d_max_)
        self.d_hat = tf.Variable(self.d_)
        self.d_de_nom = tf.Variable(0.0)
        self.d_numerator = tf.Variable(0.0)
        self._track_variable(self.d)
        self._track_variable(self.d0)
        self._track_variable(self.d_max)
        self._track_variable(self.d_hat)
        self._track_variable(self.d_de_nom)
        self._track_variable(self.d_numerator)
        self.step = 0
        for var in var_list:
            self.p0.append(tf.Variable(var))
            self._track_variable(self.p0[-1])
            self.s.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="s"
                )
            )
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
            
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        self.step += 1
        
        beta3 = self.beta3 if self.beta3 is not None else math.sqrt(self.beta2)
        
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2_sq = math.sqrt(1 - self.beta2 ** self.step)
        bias_correction = (bias_correction1 / bias_correction2_sq) if self.bias_correction else 1.0
        
        d_lr = self.d * self.lr / bias_correction
        
        self.d_numerator.assign(self.d_numerator * beta3)
        
        for variable, gradient in zip(trainable_variables, grads):
            d_lr = tf.cast(d_lr, variable.dtype)
            
            if tf.keras.backend.is_sparse(gradient):
                raise RuntimeError(
                    'Prodigy does not support sparse gradients')
            
            p0 = self.p0[self._get_variable_index(variable)]
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
            
            grad_flat = tf.reshape(gradient, [-1])
            diff_flat = tf.reshape(p0 - variable, [-1])
            self.d_numerator.assign_add(tf.cast((self.d / self.d0) * d_lr * tf.tensordot(grad_flat, diff_flat, axes=1), tf.float32))
            
            exp_avg.assign(exp_avg * self.beta1 + gradient * self.d * (1.0 - self.beta1))
            exp_avg_sq.assign(exp_avg_sq * self.beta2 + gradient * gradient * self.d * self.d * (1.0 - self.beta2))
            
            s = self.s[self._get_variable_index(variable)]
            s.assign(s * beta3 + gradient * (self.d / self.d0) * (self.d if self.safeguard_warmup else d_lr))
            
            self.d_de_nom.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
        
        def update_fn():
            d_hat = self.d
            
            if self.lr > 0.0:
                d_hat = self.d_coef * self.d_numerator / self.d_de_nom
                d = tf.cond(self.d == self.d0, lambda: tf.maximum(self.d, d_hat), lambda: self.d)
        
                d_max = tf.maximum(self.d_max, d_hat)
                d = tf.minimum(d_max, d * self.growth_rate)
            
            self.d.assign(d)
            self.d_max.assign(d_max)
            self.d_hat.assign(d_hat)
            
            for variable, gradient in zip(trainable_variables, grads):
                exp_avg = self.exp_avg[self._get_variable_index(variable)]
                exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
                
                if self.weight_decouple:
                    variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else self.lr)))
                elif self.weight_decay > 0.0:
                    gradient += variable * self.weight_decay
                    
                de_nom = tf.sqrt(exp_avg_sq)
                
                if self.epsilon is not None:
                    de_nom += d * self.epsilon
                    variable.assign_add(-d_lr * (exp_avg / de_nom))
                else:
                    update = tf.atan2(exp_avg, de_nom)
                    variable.assign_add(update * -d_lr)
        
        def no_update_fn():
            pass
        
        tf.cond(self.d_de_nom == 0, no_update_fn, update_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "beta3": self.beta3,
                "epsilon": self.epsilon,
                "d_": self.d_,
                "d0_": self.d0_,
                "d_max_": self.d_max_,
                "d_coef": self.d_coef,
                "growth_rate": self.growth_rate,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "bias_correction": self.bias_correction,
                "safeguard_warmup": self.safeguard_warmup,
                "step": self.iterations.numpy(),
            }
        )
        return config
    
    def _update_step(self):
        if hasattr(self, 'step'):
            if type(self.step) == list:
                self.step = [self.iterations.numpy() for _ in range(len(self.step))]
            else:
                self.step = self.iterations.numpy()
	
    def _apply_weight_decay(self, variables):
        pass