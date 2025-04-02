""" DAdaptAdan
https://arxiv.org/abs/2301.07733

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class DAdaptAdan(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        beta1=0.98,
        beta2=0.92,
        beta3=0.99,
        epsilon=1e-8,
        weight_decay=0.0,
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
        name="dadaptadan",
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
        self.beta3 = beta3
        self.epsilon = epsilon
        self.d0 = d0
        self.growth_rate = growth_rate
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self.self.step = 0
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
            self.exp_avg_diff[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_diff"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.s = []
        self.exp_avg = []
        self.exp_avg_sq = []
        self.exp_avg_diff = []
        self.previous_grad = []
        self.self.step = 0
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
            self.exp_avg_diff.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_diff"
                                                    ))
            self.previous_grad.append(None)
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = learning_rate
        
        d_lr = float(self.d0 * lr)
        
        g_sq = tf.convert_to_tensor([0.0])
        sk_sq_weighted = tf.convert_to_tensor([0.0])
        sk_l1 = tf.convert_to_tensor([0.0])
        
        if self.self.step == 0:
            self.gsq_weighted = tf.convert_to_tensor([0.0])
            
        for var, grad in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(grad):
                raise RuntimeError(
                    'DAdaptAdan does not support sparse gradients')
            
            if self.self.step == 0:
                self.previous_grad[self._get_variable_index(var)] = tf.Variable(grad)
                self._track_variable(self.previous_grad[self._get_variable_index(var)])
                
            grad_diff = self.previous_grad[self._get_variable_index(var)]
            self.previous_grad[self._get_variable_index(var)] += grad
            
            exp_avg = self.exp_avg[self._get_variable_index(var)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(var)]
            exp_avg_diff = self.exp_avg_diff[self._get_variable_index(var)]
            
            d_lr = tf.cast(d_lr, dtype=var.dtype)
            
            exp_avg.assign(exp_avg * self.beta1 + grad * d_lr * (1.0 - self.beta1))
            exp_avg_diff.assign(exp_avg_diff * self.beta2 + grad_diff * d_lr * (1.0 - self.beta2))
            
            self.previous_grad[self._get_variable_index(var)] = grad_diff * self.beta2 + grad
            x = grad_diff * tf.math.conj(grad_diff)
            grad_diff = tf.math.real(x) if x.dtype.is_complex else x
            exp_avg_sq.assign(exp_avg_sq * self.beta3 + grad_diff * grad_diff * (1.0 - self.beta3))
            
            x = grad * tf.math.conj(grad)
            grad_power = tf.math.real(x) if x.dtype.is_complex else x
            de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
            
            g_sq += tf.reduce_sum(grad_power / de_nom)
            
            s = self.s[self._get_variable_index(var)]
            s.assign(s * self.beta3 + grad * d_lr * (1.0 - self.beta3))
            
            x = s * tf.math.conj(s)
            x = tf.math.real(x) if x.dtype.is_complex else x
            sk_sq_weighted += tf.reduce_sum(x / de_nom)
            sk_l1 += tf.reduce_sum(tf.abs(s))
            
            self.previous_grad[self._get_variable_index(var)] = -grad
        
        if tf.get_static_value(sk_l1) == 0:
            return
        
        self.gsq_weighted = self.gsq_weighted * self.beta3 + g_sq * (d_lr ** 2) * (1.0 - self.beta3)  # fmt: skip
        
        if tf.get_static_value(lr) > 0.0:
            d_hat = (sk_sq_weighted / (1.0 - self.beta3) - self.gsq_weighted) / sk_l1
            d = max(self.d0, min(d_hat, self.d0 * self.growth_rate))
        
        self.d0 = d
        
        for var, grad in zip(trainable_variables, grads):
            exp_avg = self.exp_avg[self._get_variable_index(var)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(var)]
            exp_avg_diff = self.exp_avg_diff[self._get_variable_index(var)]
            
            de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
            
            d_lr = tf.cast(d_lr, dtype=var.dtype)
            
            if self.weight_decouple:
                var.assign(var * (1.0 - d_lr * self.weight_decay))
            
            var.assign_add(-1.0 * exp_avg / de_nom)
            var.assign_add(-self.beta2 * exp_avg_diff / de_nom)
            
            if not self.weight_decouple:
                var.assign(var / (1.0 + d_lr * self.weight_decay))
        
        self.self.step += 1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "beta3": self.beta3,
                "epsilon": self.epsilon,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "gsq_weighted": self.gsq_weighted,
                "self.step": self.self.step,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass