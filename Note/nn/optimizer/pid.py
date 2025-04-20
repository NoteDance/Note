""" PID
Implements stochastic gradient descent (optionally with momentum).
Nesterov momentum is based on the formula from
`On the importance of initialization and momentum in deep learning`__.
http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class PID(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate,
        weight_decay=0,
        momentum=0,
        dampening=0,
        nesterov=False,
        I=5.,
        D=10.,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="pid",
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
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.I = I
        self.D = D
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.nesterov = False

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.I_buffer = []
        self.grad_buffer = []
        self.D_buffer = []
        self.step = []
        for var in var_list:
            self.I_buffer.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="I_buffer"
                )
            )
            self.grad_buffer.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="grad_buffer"
                )
            )
            self.D_buffer.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="D_buffer"
                )
            )
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        d_p = gradient
        if self.weight_decay != 0:
            d_p += self.weight_decay * variable
        if self.momentum != 0:
            if self.step == 0:
                I_buf = self.I_buffer[self._get_variable_index(variable)]
                I_buf.assign(I_buf * self.momentum + d_p)
            else:
                I_buf = self.I_buffer[self._get_variable_index(variable)]
                I_buf.assign(I_buf * self.momentum + (1 - self.dampening) * d_p)
            if self.step == 0:
                g_buf = self.grad_buffer[self._get_variable_index(variable)]
                g_buf = d_p   
                
                D_buf = self.D_buffer[self._get_variable_index(variable)]
                D_buf.assign(D_buf * self.momentum + (d_p-g_buf))
            else:
                D_buf = self.D_buffer[self._get_variable_index(variable)]
                g_buf = self.grad_buffer[self._get_variable_index(variable)]                                   
                D_buf.assign(D_buf * self.momentum + (1-self.momentum) * (d_p-g_buf))   
                self.grad_buffer[self._get_variable_index(variable)] = d_p
                
            d_p = d_p + self.I * I_buf + self.D * D_buf
        
        variable.assign_add(-lr * d_p)
        
        self.step[self._get_variable_index(variable)] += 1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "dampening": self.dampening,
                "nesterov": self.nesterov,
                "I": self.I,
                "D": self.D,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass