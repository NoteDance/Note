""" Nvidia NovoGrad Optimizer.
Original impl by Nvidia from Jasper example:
    - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper
Paper: `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks`
    - https://arxiv.org/abs/1905.11286
    
Copyright 2024 NoteDance
"""

import tensorflow as tf
from keras.src.optimizers import optimizer


class NvNovoGrad(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.95,
        beta2=0.98,
        epsilon=1e-8,
        weight_decay=0,
        grad_averaging=False,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="nvnovograd",
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
        self.grad_averaging = grad_averaging
        self.amsgrad = amsgrad

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.max_exp_avg_sq = []
        var_ = tf.zeros([])
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var_, name="exp_avg_sq"
                )
            )
            if self.amsgrad:
                self.max_exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var_, name="exp_avg_sq"
                    )
                )
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.amsgrad = False

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        if self.amsgrad:
            max_exp_avg_sq = self.max_exp_avg_sq[self._get_variable_index(variable)]
        beta1, beta2 = self.beta1, self.beta2

        norm = tf.reduce_sum(tf.square(gradient))
        
        if exp_avg_sq == 0:
            exp_avg_sq.assign(norm)
        else:
            exp_avg_sq.assign(beta2 * exp_avg_sq + (1 - beta2) * norm)
            
        if self.amsgrad:
            max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq))
            denom = tf.sqrt(max_exp_avg_sq) + self.epsilon
        else:
            denom = tf.sqrt(exp_avg_sq) + self.epsilon
    
        gradient = gradient / denom
        if self.weight_decay != 0:
            gradient += self.weight_decay * variable
        if self.grad_averaging:
            gradient = gradient * (1 - beta1)
        exp_avg.assign(beta1 * exp_avg + gradient)
        
        variable.assign_add(-lr * exp_avg)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "grad_averaging": self.grad_averaging,
                "amsgrad": self.amsgrad,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass