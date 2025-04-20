""" Apollo
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import numpy as np


class Apollo(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate,
        beta=0.9,
        epsilon=1e-4,
        weight_decay=0,
        rebound='constant',
        warmup=500,
        init_lr=None,
        weight_decay_type=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="apollo",
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
        self.beta = beta
        self.epsilon = epsilon
        self.rebound = rebound
        self.warmup = warmup
        self.init_lr = init_lr
        self.weight_decay_type = weight_decay_type

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg_grad = []
        self.approx_hessian = []
        self.update = []
        self.base_lr = self._learning_rate
        self.step = []
        for var in var_list:
            self.exp_avg_grad.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_grad"
                )
            )
            self.approx_hessian.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="approx_hessian"
                )
            )
            self.update.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="update"
                )
            )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        # Calculate current lr
        if self.step[self._get_variable_index(variable)] < self.warmup:
            curr_lr = (self.base_lr - self.init_lr) * self.step[self._get_variable_index(variable)] / self.warmup + self.init_lr
            curr_lr = tf.cast(curr_lr, variable.dtype)
        else:
            curr_lr = lr

        # Perform optimization self.step
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError('Atom does not support sparse gradients.')

        # Perform self.step weight decay
        if self.weight_decay != 0 and self.weight_decay_type == 'L2':
            gradient = gradient + variable * self.weight_decay

        exp_avg_grad = self.exp_avg_grad[self._get_variable_index(variable)]
        B = self.approx_hessian[self._get_variable_index(variable)]
        d_p = self.update[self._get_variable_index(variable)]

        bias_correction = 1 - self.beta ** self.step[self._get_variable_index(variable)]
        alpha = (1 - self.beta) / bias_correction

        # calc the diff grad
        delta_grad = gradient - exp_avg_grad
        if self.rebound == 'belief':
            rebound = tf.norm(delta_grad, ord=np.inf)
        else:
            rebound = 0.01
            eps = self.epsilon / rebound

        # Update the running average grad
        exp_avg_grad.assign_add(delta_grad * alpha)

        denom = tf.norm(d_p, ord=4) + eps
        d_p.assign(d_p / denom)
        v_sq = d_p * d_p
        delta = tf.reduce_sum(delta_grad / denom * d_p) * -alpha - tf.reduce_sum(B * v_sq)

        # Update B
        B.assign_add(v_sq * delta)

        # calc direction of parameter updates
        if self.rebound == 'belief':
            denom = tf.maximum(tf.abs(B), rebound) + (eps / alpha)
        else:
            denom = tf.abs(B)
            denom = tf.clip_by_value(denom, clip_value_min=rebound, clip_value_max=denom.dtype.max)

        d_p.assign(exp_avg_grad / denom)

        # Perform self.step weight decay
        if self.weight_decay != 0 and self.weight_decay_type != 'L2':
            if self.weight_decay_type == 'stable':
                weight_decay = self.weight_decay / tf.reduce_mean(denom)
            else:
                weight_decay = self.weight_decay
            d_p.assign_add(variable * weight_decay)

        variable.assign_add(d_p * -curr_lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "epsilon": self.epsilon,
                "rebound": self.rebound,
                "warmup": self.warmup,
                "init_lr": self.init_lr,
                "weight_decay_type": self.weight_decay_type,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass