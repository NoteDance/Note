"""RAdam Optimizer.
Implementation lifted from: https://github.com/LiyuanLucasLiu/RAdam
Paper: `On the Variance of the Adaptive Learning Rate and Beyond` - https://arxiv.org/abs/1908.03265

Copyright 2024 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class RAdam(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="radam",
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

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.buffer = [[None, None, None] for _ in range(10)]
        self.step = []
        for var in var_list:
            var = tf.cast(var, 'float32')
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
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        if variable.dtype != tf.float32:
            variable_fp32 = tf.cast(variable, 'float32')
        else:
            variable_fp32 = tf.convert_to_tensor(variable)
        lr = tf.cast(learning_rate, variable.dtype)
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        beta1, beta2 = self.beta1, self.beta2
        
        exp_avg_sq.assign(beta2 * exp_avg_sq + (1 - beta2) * tf.multiply(gradient, gradient))
        exp_avg.assign(beta1 * exp_avg + (1 - beta1) * gradient)

        self.step[self._get_variable_index(variable)] += 1
        buffered = self.buffer[int(self.step[self._get_variable_index(variable)] % 10)]
        if self.step[self._get_variable_index(variable)] == buffered[0]:
            num_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = self.step[self._get_variable_index(variable)]
            beta2_t = beta2 ** self.step[self._get_variable_index(variable)]
            num_sma_max = 2 / (1 - beta2) - 1
            num_sma = num_sma_max - 2 * self.step[self._get_variable_index(variable)] * beta2_t / (1 - beta2_t)
            buffered[1] = num_sma
            
            # more conservative since it's an approximated value
            if num_sma >= 5:
                step_size = lr * math.sqrt(
                    (1 - beta2_t) *
                    (num_sma - 4) / (num_sma_max - 4) *
                    (num_sma - 2) / num_sma *
                    num_sma_max / (num_sma_max - 2)) / (1 - beta1 ** self.step[self._get_variable_index(variable)])
            else:
                step_size = lr / (1 - beta1 ** self.step[self._get_variable_index(variable)])
            buffered[2] = step_size
        
        if self.weight_decay != 0:
            variable_fp32 += -self.weight_decay * lr * variable_fp32
        
        # more conservative since it's an approximated value
        if num_sma >= 5:
            denom = tf.sqrt(exp_avg_sq) + self.epsilon
            variable_fp32 += -step_size * exp_avg / denom
        else:
            variable_fp32 += -step_size * exp_avg
        
        variable.assign(tf.cast(variable_fp32, variable.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass