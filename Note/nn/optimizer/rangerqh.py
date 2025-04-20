""" RangerQH
Implements the QHAdam optimization algorithm `(Ma and Yarats, 2019)`_.
Along with Hinton/Zhang Lookahead.
https://arxiv.org/abs/1810.06801

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class RangerQH(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        nus=(.7, 1.0),
        k=6,
        alpha=.5,
        decouple_weight_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="rangerqh",
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
        self.nus = nus
        self.k = k
        self.alpha = alpha
        self.decouple_weight_decay = decouple_weight_decay

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.slow_buffer = []
        self.beta1_weight = 0.0
        self.beta2_weight = 0.0
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
            self.slow_buffer.append(tf.Variable(var))
            self._track_variable(self.slow_buffer[-1])
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        nu1, nu2 = self.nus
        d_p = gradient
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError("QHAdam does not support sparse gradients")
        
        if self.weight_decay != 0:
            if self.decouple_weight_decay:
                variable.assign(variable * (1 - lr * self.weight_decay))
            else:
                d_p += self.weight_decay * variable

        d_p_sq = d_p * d_p
        
        self.step[self._get_variable_index(variable)] += 1
        
        self.beta1_weight = 1.0 + self.beta1 * self.beta1_weight
        self.beta2_weight = 1.0 + self.beta2 * self.beta1_weight
        
        self.beta1_weight = self.beta1_weight
        self.beta2_weight = self.beta2_weight
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        
        beta1_adj = 1.0 - (1.0 / self.beta1_weight)
        beta2_adj = 1.0 - (1.0 / self.beta2_weight)
        exp_avg.assign(exp_avg * beta1_adj + (1.0 - beta1_adj) * d_p)
        exp_avg_sq.assign(exp_avg_sq * beta2_adj + (1.0 - beta2_adj) * d_p_sq)
        
        avg_grad = exp_avg * nu1
        if nu1 != 1.0:
            avg_grad.assign_add(1.0 - nu1 * d_p)
        
        avg_grad_rms = exp_avg_sq * nu2
        if nu2 != 1.0:
            avg_grad_rms.assign_add(1.0 - nu2 * d_p_sq)
        avg_grad_rms = tf.sqrt(avg_grad_rms)
        if self.epsilon != 0.0:
            avg_grad_rms.assign_add(self.epsilon)
        
        variable.assign_add(-lr * (avg_grad / avg_grad_rms))

        # integrated look ahead...
        # we do it at the param level instead of group level
        if self.step[self._get_variable_index(variable)] % self.k == 0:
            # get access to slow param tensor
            slow_p = self.slow_buffer[self._get_variable_index(variable)]
            # (fast weights - slow weights) * alpha
            slow_p.assign_add(self.alpha * (variable - slow_p))
            # copy interpolated weights to RAdam param tensor
            variable.assign(slow_p)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "nus": self.nus,
                "k": self.k,
                "alpha": self.alpha,
                "decouple_weight_decay": self.decouple_weight_decay,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass