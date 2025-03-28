""" Note LARS / LARC Optimizer

An implementation of LARS (SGD) + LARC in PyTorch

Based on:
  * PyTorch SGD: https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
  * NVIDIA APEX LARC: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py

Additional cleanup and modifications to properly support PyTorch XLA.

Copyright 2024 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class Lars(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        momentum=0,
        dampening=0,
        epsilon=1e-8,
        weight_decay=0,
        nesterov=False,
        trust_coeff=0.001,
        trust_clip=False,
        always_adapt=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="lars",
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
        self.epsilon = epsilon
        self.nesterov = nesterov
        self.trust_coeff = trust_coeff
        self.trust_clip = trust_clip
        self.always_adapt = always_adapt

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.param_state = []
        for var in var_list:
            self.param_state.append(dict())

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        # apply LARS LR adaptation, LARC clipping, weight decay
        # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
        if self.weight_decay != 0 or self.always_adapt:
            w_norm = tf.norm(variable, ord=2.0)
            g_norm = tf.norm(gradient, ord=2.0)
            trust_ratio = self.trust_coeff * w_norm / (g_norm + w_norm * self.weight_decay + self.epsilon)
            # FIXME nested where required since logical and/or not working in PT XLA
            # Set the ratio to 1.0 (no change) if either weight norm or grad norm is zero
            trust_ratio = tf.where(
                w_norm > 0,
                tf.where(g_norm > 0, trust_ratio, 1.0),
                1.0,
            )
            if self.trust_clip:
                trust_ratio = tf.clip_by_value(trust_ratio / lr, clip_value_min=-float('inf'), clip_value_max=1.0)
            gradient += variable * self.weight_decay
            gradient = gradient * trust_ratio

        # apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
        if self.momentum != 0:
            if 'momentum_buffer' not in self.param_state[self._get_variable_index(variable)]:
                buf = self.param_state[self._get_variable_index(variable)]['momentum_buffer'] = gradient
            else:
                buf = self.param_state[self._get_variable_index(variable)]['momentum_buffer']
                self.param_state[self._get_variable_index(variable)]['momentum_buffer'] = buf * self.momentum + gradient * (1. - self.dampening)
            if self.nesterov:
                gradient = gradient + buf * self.momentum
            else:
                gradient = buf

        variable.assign_add(gradient * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "dampening": self.dampening,
                "epsilon": self.epsilon,
                "nesterov": self.nesterov,
                "trust_coeff": self.trust_coeff,
                "trust_clip": self.trust_clip,
                "always_adapt": self.always_adapt,
                "param_state": self.param_state,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass