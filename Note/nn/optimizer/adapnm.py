""" AdaPNM
https://arxiv.org/abs/2103.17182

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class AdaPNM(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        beta3=1.0,
        epsilon=1e-8,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        ams_bound=True,
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
        name="adapnm",
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
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
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
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )
            self.neg_exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="neg_exp_avg"
                                                    )
            if self.ams_bound:
                self.max_exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="max_exp_avg_sq"
                                                        )
            if self.adanorm:
                self.exp_grad_norm[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(tf.zeros((1,), dtype=var.dtype)), name="exp_grad_norm"
                                                            )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.neg_exp_avg = []
        if self.ams_bound:
            self.max_exp_avg_sq = []
        if self.adanorm:
            self.exp_grad_norm = []
        self.step = []
        for var in var_list:
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
            self.neg_exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="neg_exp_avg"
                )
            )
            if self.ams_bound:
                self.max_exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_sq"
                    )
                )
            if self.adanorm:
                self.exp_grad_norm.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.Variable(tf.zeros((1,), dtype=var.dtype)), name="exp_grad_norm"
                    )
                )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        noise_norm = math.sqrt((1 + self.beta3) ** 2 + self.beta3 ** 2)  # fmt: skip
        
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2_sq = math.sqrt(1 - self.beta2 ** self.step[self._get_variable_index(variable)])
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdaPNM does not support sparse gradients')
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        if self.step[self._get_variable_index(variable)] % 2 == 1:
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            neg_exp_avg = self.neg_exp_avg[self._get_variable_index(variable)]
        else:
            exp_avg = self.neg_exp_avg[self._get_variable_index(variable)]
            neg_exp_avg = self.exp_avg[self._get_variable_index(variable)]
        
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
        
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta1 ** 2 + s_grad * (1.0 - self.beta1 ** 2))  # fmt: skip
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + (1 - self.beta2) * tf.square(gradient))
        
        if self.ams_bound:
            max_exp_avg_sq = self.max_exp_avg_sq[self._get_variable_index(variable)]
            max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq))
            de_nom = max_exp_avg_sq + self.epsilon
        else:
            de_nom = exp_avg_sq + self.epsilon
        de_nom = tf.sqrt(de_nom) + self.epsilon
        de_nom /= bias_correction2_sq
        
        step_size = lr if self.adam_debias else lr / bias_correction1
        
        pn_momentum = exp_avg * (1.0 + self.beta3) + neg_exp_avg * -self.beta3 * (1.0 / noise_norm)
        variable.assign_add(-step_size * (pn_momentum / de_nom))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "beta3": self.beta3,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
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