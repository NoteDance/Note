""" AdEMAMix
https://arxiv.org/abs/2409.03137

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class AdEMAMix(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        beta3=0.9999,
        epsilon=1e-8,
        weight_decay=0,
        alpha=5.0,
        T_alpha_beta3=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="ademamix",
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
        self.alpha = alpha
        self.T_alpha_beta3 = T_alpha_beta3

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.exp_avg_slow = []
        self.step = 0
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
            self.exp_avg_slow.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_slow"
                )
            )
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        exp_avgs = []
        exp_avg_sqs = []
        exp_avg_slow = []
        state_steps = []
        
        self.step += 1
        
        for p in trainable_variables:
            exp_avgs.append(self.exp_avg[self._get_variable_index(p)])
            exp_avg_sqs.append(self.exp_avg_sq[self._get_variable_index(p)])
            exp_avg_slow.append(self.exp_avg_slow[self._get_variable_index(p)])
            state_steps.append(self.step)
        
        self._update_adamemix(
            trainable_variables,
            grads,
            exp_avgs,
            exp_avg_sqs,
            exp_avg_slow,
            state_steps,
            beta1=self.beta1,
            beta2=self.beta2,
            beta3=self.beta3,
            alpha=self.alpha,
            T_alpha_beta3=self.T_alpha_beta3,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.epsilon,
        )
    
    def _update_adamemix(self, params, grads, exp_avgs, exp_avg_sqs, exp_avg_slow, state_steps,
                         beta1, beta2, beta3, alpha, T_alpha_beta3, lr, weight_decay, eps):
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_slow_i = exp_avg_slow[i]
            self.step = state_steps[i]

            bias_correction1 = 1 - beta1 ** self.step
            bias_correction2 = 1 - beta2 ** self.step

            if T_alpha_beta3 is not None:
                alpha_t = min(self.step * alpha / T_alpha_beta3, alpha)
                beta3_t = min(math.exp(math.log(beta1) * math.log(beta3) / 
                              ((1 - self.step / T_alpha_beta3) * math.log(beta3) + 
                               (self.step / T_alpha_beta3) * math.log(beta1))), beta3)
            else:
                alpha_t = alpha
                beta3_t = beta3

            # Decay the first and second moment running average coefficient
            exp_avg.assign(exp_avg * beta1 + grad * (1 - beta1))
            exp_avg_sq.assign(exp_avg_sq * beta2 + grad * grad * (1 - beta2))
            exp_avg_slow_i.assign(exp_avg_slow_i * beta3_t + grad * (1 - beta3_t))

            denom = tf.sqrt(exp_avg_sq) / math.sqrt(bias_correction2) + eps

            step_size = lr / bias_correction1

            if weight_decay != 0:
                param.assign_add(param * -weight_decay * lr)

            param.assign_add(-step_size * (exp_avg + alpha_t * exp_avg_slow_i / denom))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "beta3": self.beta3,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "T_alpha_beta3": self.T_alpha_beta3,
                "step": self.step,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass