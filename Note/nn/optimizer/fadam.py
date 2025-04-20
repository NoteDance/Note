""" FAdam
https://arxiv.org/abs/2405.12807

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class FAdam(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.1,
        clip=1.0,
        p=0.5,
        momentum_dtype=tf.float32,
        fim_dtype=tf.float32,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="fadam",
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
        self.clip = clip
        self.p = p
        self.momentum_dtype = momentum_dtype
        self.fim_dtype = fim_dtype
    
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
            self.momentum[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=tf.Variable(tf.cast(var, dtype=self.momentum_dtype)), name="momentum"
                                                    )
            self.fim[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=tf.Variable(tf.cast(var, dtype=self.fim_dtype)), name="fim"
                                                    )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum = []
        self.fim = []
        self.step = []
        for var in var_list:
            self.momentum.append(self.add_variable_from_reference(
                                reference_variable=tf.Variable(tf.cast(var, dtype=self.momentum_dtype)), name="momentum"
                                                    ))
            self.fim.append(self.add_variable_from_reference(
                                reference_variable=tf.Variable(tf.cast(var, dtype=self.fim_dtype)), name="fim"
                                                    ))
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        curr_beta2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'FAdam does not support sparse gradients')
        
        momentum = self.momentum[self._get_variable_index(variable)]
        fim = self.fim[self._get_variable_index(variable)]
        fim.assign(fim * curr_beta2 + gradient * gradient * (1.0 - curr_beta2))
        
        rms_grad = tf.sqrt(tf.reduce_mean(tf.pow(gradient, 2)))
        curr_eps = tf.minimum(rms_grad, tf.cast(1, rms_grad.dtype)) * self.epsilon
        
        fim_base = tf.pow(fim, self.p) + curr_eps
        grad_nat = gradient / tf.cast(fim_base, dtype=variable.dtype)
        
        rms = tf.sqrt(tf.reduce_mean(tf.pow(grad_nat, 2)))
        divisor = tf.maximum(tf.cast(1, rms.dtype), rms) / self.clip
        grad_nat = grad_nat / divisor
        
        momentum.assign(momentum * self.beta1 + grad_nat * (1.0 - self.beta1))
        
        grad_weights = self.p / fim_base

        rms = tf.sqrt(tf.reduce_mean(tf.pow(grad_weights, 2)))
        divisor = tf.maximum(tf.cast(1, rms.dtype), rms) / self.clip
        grad_weights = grad_weights / divisor

        grad_weights = grad_weights * self.weight_decay + momentum

        variable.assign_add(grad_weights * lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "clip": self.clip,
                "p": self.p,
                "momentum_dtype": self.momentum_dtype,
                "fim_dtype": self.fim_dtype,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass