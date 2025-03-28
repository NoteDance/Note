""" AliG
https://arxiv.org/abs/1906.05661

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def l2_projection(parameters, max_norm = 1e2):
    r"""Get l2 normalized parameter."""
    global_norm = tf.sqrt(sum(tf.pow(tf.norm(p), 2) for p in parameters))
    if global_norm > max_norm:
        ratio = max_norm / global_norm
        for param in parameters:
            param.assign(param * ratio)


def get_global_gradient_norm(grads):
    r"""Get global gradient norm."""
    global_grad_norm = tf.zeros(1, dtype=tf.float32)

    for g in grads:
        global_grad_norm += tf.pow(tf.norm(g), 2)

    return global_grad_norm


class AliG(optimizer.Optimizer):
    def __init__(
        self,
        max_lr=None,
        projection_fn=None,
        momentum=0.0,
        adjusted_momentum=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="alig",
        **kwargs,
    ):
        super().__init__(
            learning_rate=1.,
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
        self.max_lr = max_lr
        self.projection_fn = projection_fn
        self.momentum = momentum
        self.adjusted_momentum = adjusted_momentum
        
        if self.projection_fn is not None:
            self.projection_fn()
    
    def reset(self):
        for var in self._trainable_variables:
            if self.momentum > 0.0:
                self.momentum_buffer[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="momentum_buffer"
                                                        )
    
    def compute_step_size(self):
        r"""Compute step_size."""
        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm += 1e-6

        return self.loss / global_grad_norm

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        if self.momentum > 0.0:
            self.momentum_buffer = []
        for var in var_list:
            if self.momentum > 0.0:
                self.momentum_buffer.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="momentum_buffer"
                    )
                )
    
    def apply_gradients(self, grads_and_vars, loss):
        self.loss = loss
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        un_clipped_step_size = tf.get_static_value(self.compute_step_size())
        
        step_size = (
            min(un_clipped_step_size, self.max_lr) if self.max_lr is not None else un_clipped_step_size
        )
        
        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'AliG does not support sparse gradients')

            p.assign_add(g * -step_size)

            if self.momentum > 0.0:
                buffer = self.momentum_buffer[self._get_variable_index(p)]

                if self.adjusted_momentum:
                    buffer.assign(buffer * self.momentum - g)
                    p.assign_add(buffer * step_size * self.momentum)
                else:
                    buffer.assign(buffer * self.momentum + g * -step_size)
                    p.assign_add(buffer * self.momentum)

        if self.projection_fn is not None:
            self.projection_fn()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_lr": self.max_lr,
                "projection_fn": self.projection_fn,
                "momentum": self.momentum,
                "adjusted_momentum": self.adjusted_momentum,
            }
        )
        return config