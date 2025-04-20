""" Aida
https://arxiv.org/abs/2203.13273

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class Aida(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        k=2,
        xi=1e-20,
        weight_decouple=False,
        fixed_decay=False,
        rectify=False,
        n_sma_threshold=5,
        degenerated_to_sgd=True,
        ams_bound=False,
        r=0.95,
        adanorm=False,
        adam_debias=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="aida",
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
        self.k = k
        self.xi = xi
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify
        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.ams_bound = ams_bound
        self.r = r
        self.adanorm = adanorm
        self.adam_debias = adam_debias
    
    def reset(self):
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
            
            self.exp_avg_var[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_var"
                                                    )
            
            if self.adanorm:
                self.exp_grad_norm[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=tf.Variable(tf.zeros((1,), dtype=var.dtype)), name="exp_grad_norm"
                                                    )
            
            if self.ams_bound:
                self.max_exp_avg_var[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="max_exp_avg_var"
                                                    )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_var = []
        if self.adanorm:
            self.exp_grad_norm = []
        if self.ams_bound:
            self.max_exp_avg_var = []
        self.step = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_var.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_var"
                )
            )
            if self.adanorm:
                self.exp_grad_norm.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.Variable(tf.zeros((1,), dtype=var.dtype)), name="exp_grad_norm"
                    )
                )
            if self.ams_bound:
                self.max_exp_avg_var.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_var"
                    )
                )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Aida does not support sparse gradients')
        
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2_sq = math.sqrt(1 - self.beta2 ** self.step[self._get_variable_index(variable)])
        
        step_size = lr
        n_sma = 0.0
        
        if self.rectify:
            n_sma_max = 2.0 / (1.0 - self.beta2) - 1.0
            beta2_t = self.beta2 ** self.step[self._get_variable_index(variable)]  # fmt: skip
            n_sma = n_sma_max - 2 * self.step[self._get_variable_index(variable)] * beta2_t / (1.0 - beta2_t)
        
            if n_sma >= self.n_sma_threshold:
                rt = math.sqrt(
                    (1.0 - beta2_t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma * n_sma_max / (n_sma_max - 2)
                )
            elif self.degenerated_to_sgd:
                rt = 1.0
            else:
                rt = -1.0
        
            step_size *= rt
        
        step_size = step_size if self.adam_debias else lr / bias_correction1
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        if self.adanorm:
            grad_norm = tf.linalg.norm(gradient)
            exp_grad_norm = self.exp_grad_norm[self._get_variable_index(variable)]
            exp_grad_norm.assign(exp_grad_norm * self.r + grad_norm * (1.0 - self.r))
            def true_fn():
                return gradient * exp_grad_norm / grad_norm
            def false_fn():
                return gradient
            s_grad = tf.cond(exp_grad_norm > grad_norm, true_fn, false_fn)
        else:
            s_grad = gradient
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_var = self.exp_avg_var[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta1 + s_grad * (1.0 - self.beta1))
                       
        proj_g = gradient
        proj_m = exp_avg
        
        for _ in range(self.k):
            proj_sum_gm = tf.reduce_sum(proj_g * proj_m)

            scalar_g = proj_sum_gm / (tf.reduce_sum(tf.pow(proj_g, 2)) + self.xi)
            scalar_m = proj_sum_gm / (tf.reduce_sum(tf.pow(proj_m, 2)) + self.xi)

            proj_g = proj_g * scalar_g
            proj_m = proj_m * scalar_m
      
        grad_residual = proj_m - proj_g
        exp_avg_var.assign(
            exp_avg_var * self.beta2 + (grad_residual * grad_residual) * (1.0 - self.beta2) + self.epsilon
        )
        
        if self.ams_bound:
            max_exp_avg_var = self.max_exp_avg_var[self._get_variable_index(variable)]
            max_exp_avg_var.assign(tf.maximum(max_exp_avg_var, exp_avg_var))
            de_nom = max_exp_avg_var + self.epsilon
        else:
            de_nom = exp_avg_var + self.epsilon
        de_nom = tf.sqrt(de_nom) + self.epsilon
        
        if not self.rectify:
            de_nom /= bias_correction2_sq
            variable.assign_add(-step_size * (exp_avg / de_nom))
            
        if n_sma >= self.n_sma_threshold:
            variable.assign_add(-step_size * (exp_avg / de_nom))
        elif step_size > 0:
            variable.assign_add(-step_size * exp_avg)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "k": self.k,
                "xi": self.xi,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "rectify": self.rectify,
                "n_sma_threshold": self.n_sma_threshold,
                "degenerated_to_sgd": self.degenerated_to_sgd,
                "ams_bound": self.ams_bound,
                "r": self.r,
                "adanorm": self.adanorm,
                "adam_debias": self.adam_debias,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass