""" AdaGC
https://arxiv.org/abs/2502.11034

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AdaGC(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        beta=0.98,
        epsilon=1e-8,
        weight_decay=1e-1,
        lambda_abs=1.0,
        lambda_rel=1.05,
        warmup_steps=100,
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
        name="adagc",
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
        self.beta = beta
        self.epsilon = epsilon
        self.lambda_abs = lambda_abs
        self.lambda_rel = lambda_rel
        self.warmup_steps = warmup_steps
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )
            self.gamma[self._get_variable_index(var)].assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.gamma = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.gamma.append(tf.Variable(tf.zeros((), dtype=var.dtype)))
            self._track_variable(self.gamma[-1])
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        beta1, beta2 = self.betas
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'AdaGC does not support sparse gradients')
            
            lr = tf.cast(learning_rate, p.dtype)
            
            step = tf.cast(self.iterations + 1, p.dtype)
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2_sq = tf.sqrt(1 - beta2 ** step)
            
            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * self.weight_decay
                
            gamma = self.gamma[self._get_variable_index(p)]
            
            def true_fn():
                global_grad_norm = tf.zeros((), dtype=tf.float32)
                for g in grads:
                    norm_g = tf.norm(g)
                    global_grad_norm += tf.square(norm_g)
                grad_norm = global_grad_norm + self.epsilon
    
                h_t = tf.minimum(self.lambda_abs / grad_norm, 1.0)
                g_hat = g * h_t
    
                g_hat_norm = tf.norm(g_hat)
                
                def true_fn():
                    gamma.assign(g_hat_norm)
                def false_fn():
                    gamma.assign(tf.minimum(gamma, g_hat_norm))
                tf.cond(step == 1, true_fn, false_fn)
                
                return g_hat
            
            def false_fn():
                h_t = tf.minimum(self.lambda_rel * gamma / tf.norm(g), 1.0)
                g_hat = g * h_t

                gamma.assign( gamma * self.beta + tf.norm(g_hat) * (1.0 - self.beta))
                
                return g_hat
            
            g_hat = tf.cond(step < self.warmup_steps, true_fn, false_fn)
            
            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
            exp_avg.assign(exp_avg * beta1 + g_hat * (1.0 - beta1))
            exp_avg_sq.assign(exp_avg_sq * beta2 + g_hat * g_hat * (1.0 - beta2))
    
            update = (exp_avg / bias_correction1) / tf.sqrt(exp_avg_sq) / bias_correction2_sq + self.epsilon

            p.assign_add(update * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "beta": self.beta,
                "epsilon": self.epsilon,
                "lambda_abs": self.lambda_abs,
                "lambda_rel": self.lambda_rel,
                "warmup_steps": self.warmup_steps,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
