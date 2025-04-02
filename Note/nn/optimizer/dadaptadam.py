""" DAdaptAdam
https://arxiv.org/abs/2301.07733

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class DAdaptAdam(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        d0=1e-6,
        growth_rate=float('inf'),
        weight_decouple=True,
        fixed_decay=False,
        bias_correction=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="dadaptadam",
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.d0 = d0
        self.growth_rate = growth_rate
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.bias_correction = bias_correction
    
    def reset(self):
        self.step = 0
        for var in self._trainable_variables:
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="s"
                                                    )
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
        self.exp_avg = []
        self.exp_avg_sq = []
        self.numerator_weighted = None
        self.step = 0
        for var in var_list:
            self.s.append(self.add_variable_from_reference(
                                reference_variable=var, name="s"
                                                    ))
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = learning_rate
        
        self.step += 1
        
        beta2_sq = math.sqrt(self.beta2)
        
        bias_correction1 = 1.0 - self.beta1 ** (self.step)
        bias_correction2_sq = math.sqrt(1.0 - self.beta2 ** (self.step))
        bias_correction = bias_correction1 / bias_correction2_sq
        
        # it's not Adam Debias
        d_lr = self.d0 * lr if not self.bias_correction else self.d0 * lr / bias_correction
        
        sk_l1 = tf.convert_to_tensor([0.0])
        numerator_acc = tf.convert_to_tensor([0.0])
        
        if self.numerator_weighted == None:
            self.numerator_weighted = tf.convert_to_tensor([0.0])
        
        for variable, gradient in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(gradient):
                raise RuntimeError(
                    'DAdaptAdam does not support sparse gradients')
            
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
            s = self.s[self._get_variable_index(variable)]
            
            de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
            flat_grad = tf.reshape(gradient, [-1])
            flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
            dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
            numerator_acc = d_lr * dot_val
            
            d_lr = tf.cast(d_lr, dtype=variable.dtype)
            exp_avg.assign(exp_avg * self.beta1 + gradient * d_lr * (1.0 - self.beta1))
            exp_avg_sq.assign(exp_avg_sq * self.beta2 + gradient * gradient * (1.0 - self.beta2))
            
            s.assign(s * beta2_sq + gradient * d_lr * (1.0 - beta2_sq))
            
            sk_l1.assign_add(tf.reduce_sum(tf.abs(s)))
        
        if tf.get_static_value(sk_l1) == 0:
            return
        
        self.numerator_weighted = self.numerator_weighted * beta2_sq + numerator_acc * (1.0 - beta2_sq)  # fmt: skip
        
        if tf.get_static_value(lr) > 0.0:
            d_hat = self.numerator_weighted / (1.0 - beta2_sq) * sk_l1
            d = max(self.d0, min(tf.get_static_value(d_hat), self.d0 * self.growth_rate))
        
        self.d0 = d
        
        for variable, gradient in zip(trainable_variables, grads):
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
            
            de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            
            variable.assign_add(-1.0 * (exp_avg / de_nom))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "bias_correction": self.bias_correction,
                "numerator_weighted": self.numerator_weighted,
                "step": self.step,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass