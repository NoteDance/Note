""" FOCUS
https://arxiv.org/abs/2501.12243

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class FOCUS(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        gamma=0.1,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="focus",
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
        self.gamma = gamma
    
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
            self.pbar[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="pbar"
                                                    )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.pbar = []
        self.step = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.pbar.append(self.add_variable_from_reference(
                                reference_variable=var, name="pbar"
                                                    ))
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'FOCUS does not support sparse gradients')
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        pbar = self.pbar[self._get_variable_index(variable)]
        
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
        pbar.assign(pbar * self.beta2 + variable * (1.0 - self.beta2))
        
        pbar_hat = pbar / bias_correction2
        
        if self.weight_decay > 0.0:
            variable.assign_add(pbar_hat * -lr * self.weight_decay)
        
        update = tf.sign(variable - pbar_hat) * self.gamma + tf.sign(exp_avg)

        variable.assign_add(update * -lr)
        

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "gamma": self.gamma,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass