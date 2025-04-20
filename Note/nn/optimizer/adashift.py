""" AdaShift
https://arxiv.org/abs/1810.00143

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from collections import deque


class AdaShift(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-10,
        keep_num=10,
        reduce_func=tf.reduce_max,
        cautious=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adashift",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=None,
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
        self.keep_num = keep_num
        self.reduce_func = reduce_func if reduce_func is not None else lambda x: x
        self.cautious = cautious

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.grad_deque = []
        self.exp_avg = []
        self.exp_avg_sq = []
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
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        exp_weight_sum = sum(self.beta1 ** i for i in range(self.keep_num))  # fmt: skip
        first_grad_weight = self.beta1 ** (self.keep_num - 1) / exp_weight_sum
        last_grad_weight = 1.0 / exp_weight_sum
        
        bias_correction = 1 - self.beta1 ** (self.step[self._get_variable_index(variable)] - self.keep_num)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdaShift does not support sparse gradients')
        
        if self.step[self._get_variable_index(variable)] == 1:
            self.grad_deque.append(deque([gradient], maxlen=self.keep_num))
            self._track_variable(self.grad_deque[self._get_variable_index(variable)][-1])
        
        grad_deque = self.grad_deque[self._get_variable_index(variable)]
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        
        grad_apply = len(grad_deque) == self.keep_num
        offset_grad = grad_deque[0]
        grad_deque.append(gradient)
        if not grad_apply:
          return
      
        exp_avg.assign((exp_avg - first_grad_weight * offset_grad) * self.beta1 + last_grad_weight * gradient)
        
        reduced_grad_sq = self.reduce_func(offset_grad.assign(offset_grad * offset_grad))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + (1 - self.beta2) * reduced_grad_sq)
        
        update = exp_avg
        if self.cautious:
            mask = tf.cast(tf.math.greater(update * gradient, 0), gradient.dtype)
            numel = tf.cast(tf.size(mask), gradient.dtype)
            factor = numel / (tf.reduce_sum(mask) + 1)
            mask = mask * factor
            update = update * mask
            
        update /= tf.sqrt(exp_avg_sq / bias_correction) + self.epsilon
        
        variable.assign_add(-lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "keep_num": self.keep_num,
                "cautious": self.cautious,
                "grad_deque": self.grad_deque,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config