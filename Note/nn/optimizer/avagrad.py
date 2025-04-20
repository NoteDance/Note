""" AvaGrad
https://arxiv.org/abs/1912.01823

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class AvaGrad(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-1,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        adam_debias=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="avagrad",
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
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.adam_debias = adam_debias
    
    def reset(self):
        self.step = 0
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

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.gamma = tf.Variable(0.)
        self.step = 0
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        self.step += 1
        
        for variable, gradient in zip(trainable_variables, grads):
            lr = tf.cast(learning_rate, variable.dtype)
            
            bias_correction1 = 1 - self.beta1 ** self.step
            bias_correction2_sq = math.sqrt(1 - self.beta2 ** self.step)
            prev_bias_correction2_sq = math.sqrt(1 - self.beta2 ** (self.step - 1))
            
            squared_norm = tf.cast(0.0, variable.dtype)
            num_params = tf.cast(0.0, variable.dtype)
            
            if tf.keras.backend.is_sparse(gradient):
                raise RuntimeError(
                    'AvaGrad does not support sparse gradients')
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                gradient += variable * self.weight_decay
    
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
    
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
            sqrt_exp_avg_sq = tf.sqrt(exp_avg_sq)
    
            if self.step > 1:
                de_nom = sqrt_exp_avg_sq / prev_bias_correction2_sq + self.epsilon
                
                gamma = tf.cast(self.gamma, variable.dtype)
                step_size = gamma * lr if self.adam_debias else gamma * lr / bias_correction1
                variable.assign_add(-step_size * (exp_avg / de_nom))
            
            exp_avg_sq.assign(exp_avg_sq * self.beta2 + gradient * gradient * (1.0 - self.beta2))
            
            param_wise_lr = sqrt_exp_avg_sq / bias_correction2_sq + self.epsilon
            sum_power = tf.reduce_sum(tf.pow(tf.abs(param_wise_lr), -2))
            squared_norm += tf.pow(sum_power, 1.0 / -2)
            num_params += tf.size(param_wise_lr, num_params.dtype)
        
        def true_fn():
            return 0.0
    
        def false_fn():
            return tf.sqrt(squared_norm / num_params)
        
        self.gamma.assign(tf.cast(tf.cond(
            tf.equal(num_params, 0.0),
            true_fn,
            false_fn
        ), self.gamma.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "adam_debias": self.adam_debias,
                "step": self.iterations.numpy(),
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass