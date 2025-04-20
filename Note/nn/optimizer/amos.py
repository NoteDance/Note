""" Amos
https://arxiv.org/abs/2210.11693

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class Amos(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta=0.999,
        epsilon=1e-18,
        momentum=0.0,
        extra_l2=0.0,
        c_coef=0.25,
        d_coef=0.25,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="amos",
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
        self.beta = beta
        self.epsilon = epsilon
        self.momentum = momentum
        self.extra_l2 = extra_l2
        self.c_coef = c_coef
        self.d_coef = d_coef
    
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
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                        reference_variable=tf.Variable(tf.zeros((1,)), dtype=var.dtype), name="exp_avg_sq"
                                    )
            self.decay[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                        reference_variable=tf.Variable(tf.zeros((1,)), dtype=var.dtype), name="decay"
                                    )
            if self.momentum > 0.0:
                self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                            reference_variable=var, name="exp_avg"
                                        )
            self.step[self._get_variable_index(var)] = 0
    
    @staticmethod
    def get_scale(p):
        r"""Get expected scale for model weights."""
        if len(p.shape) == 1:  # expected 'bias'
            return 0.5
        if len(p.shape) == 2:  # expected Embedding, Linear, ...
            return math.sqrt(2 / p.shape[1])
        return math.sqrt(1 / p.shape[1])

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg_sq = []
        self.decay = []
        if self.momentum > 0.0:
            self.exp_avg = []
        self.step = []
        for var in var_list:
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=tf.Variable(tf.zeros((1,)), dtype=var.dtype), name="exp_avg_sq"
                )
            )
            self.decay.append(
                self.add_variable_from_reference(
                    reference_variable=tf.Variable(tf.zeros((1,)), dtype=var.dtype), name="decay"
                )
            )
            if self.momentum > 0.0:
                self.exp_avg.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="exp_avg"
                    )
                )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)

        self.step[self._get_variable_index(variable)] += 1
        
        lr_sq = math.sqrt(lr)
        bias_correction = 1 - self.beta ** self.step[self._get_variable_index(variable)]
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Amos does not support sparse gradients')
        
        g2 = tf.reduce_mean(tf.pow(gradient, 2))
        init_lr = lr * self.get_scale(variable)
        
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg_sq.assign(exp_avg_sq * self.beta + g2 * (1.0 - self.beta))
        
        r_v_hat = bias_correction / (exp_avg_sq + self.epsilon)
        
        b = self.decay[self._get_variable_index(variable)]
        decay_factor_c = tf.math.rsqrt(1.0 + self.c_coef * lr_sq * b)
        decay_factor_d = tf.math.reciprocal(1.0 + self.d_coef * math.sqrt(init_lr) * b)
        
        gamma = decay_factor_c * (lr ** 2) * r_v_hat * g2
        
        update = variable
        update = update * (gamma - self.extra_l2) / 2.0
        update += tf.sqrt(r_v_hat) * gradient * init_lr
        update = update * decay_factor_d
        
        b.assign(b * (1.0 + gamma) + gamma)
        
        if self.momentum > 0.0:
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            exp_avg.assign(exp_avg * self.momentum + update * (1.0 - self.momentum))

            update.assign(exp_avg)

        variable.assign_add(-update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "extra_l2": self.extra_l2,
                "c_coef": self.c_coef,
                "d_coef": self.d_coef,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config