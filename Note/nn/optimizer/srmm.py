""" SRMM
https://arxiv.org/abs/2201.01652

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class SRMM(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        beta=0.5,
        memory_length=100,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="srmm",
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
        self.memory_length = memory_length
    
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
            self.mov_avg_grad[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="mov_avg_grad"
                                                    )
            self.mov_avg_param[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="mov_avg_param"
                                                    )
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.mov_avg_grad = []
        self.mov_avg_param = []
        self.step = []
        for var in var_list:
            self.mov_avg_grad.append(self.add_variable_from_reference(
                                reference_variable=var, name="mov_avg_grad"
                                                    ))
            self.mov_avg_param.append(self.add_variable_from_reference(
                                reference_variable=var, name="mov_avg_param"
                                                    ))
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        w_t: float = (
            (self.step[self._get_variable_index(variable)] % (self.memory_length if self.memory_length is not None else 1)) + 1
        ) ** -self.beta
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'SRMM does not support sparse gradients')

        mov_avg_grad = self.mov_avg_grad[self._get_variable_index(variable)]
        mov_avg_param = self.mov_avg_param[self._get_variable_index(variable)]
        
        mov_avg_grad.assign(mov_avg_grad * (1.0 - w_t) + gradient * w_t)
        mov_avg_param.assign(mov_avg_param * (1.0 - w_t) + variable * w_t)

        mov_avg_param.assign_add(mov_avg_grad * -lr)

        variable.assign(mov_avg_param)      

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "memory_length": self.memory_length,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass