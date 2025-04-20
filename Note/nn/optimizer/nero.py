""" Nero
https://arxiv.org/abs/2102.07227

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def neuron_norm(x):
    r"""Get norm of the tensor."""
    if len(x.shape) <= 1:
        return tf.abs(x)

    view_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)

    return tf.reshape(tf.norm(tf.reshape(x, (x.shape[0], -1)), axis=1), *view_shape)


def neuron_mean(x):
    r"""Get mean of the tensor."""
    if len(x.shape) <= 1:
        raise ValueError('[-] neuron_mean not defined on 1D tensors.')

    view_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)

    return tf.reshape(tf.reduce_mean(tf.reshape(x, (x.shape[0], -1)), axis=1), *view_shape)


def nan_to_num(tensor, nan=0.0, out=None):
    result = tf.where(tf.math.is_nan(tensor), tf.constant(nan, dtype=tensor.dtype), tensor)
    if out is not None:
        out.assign(result)
        return out
    return result


class Nero(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        beta=0.999,
        epsilon=1e-8,
        constraints=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="nero",
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
        self.constraints = constraints
                    
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
            if self.constraints and len(var.shape) > 1:
                var.assign_sub(neuron_mean(var))
                var.assign(var / (neuron_norm(var) + self.epsilon))
                
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=tf.Variable(neuron_norm(var)), name="exp_avg_sq"
                                                    )
            var_ = tf.reduce_mean(neuron_norm(var))
            self.scale[self._get_variable_index(var)] = var_
            
            if self.scale[self._get_variable_index(var)].numpy() == 0.0:
                self.scale[self._get_variable_index(var)] = 0.01
            
            self.step[self._get_variable_index(var)] = 0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg_sq = []
        self.scale = []
        self.step = []
        for var in var_list:
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=tf.Variable(neuron_norm(var)), name="exp_avg_sq"
                                                    ))
            var_ = tf.reduce_mean(neuron_norm(var))
            self.scale.append(var_)
            def true_fn():
                self.scale[self._get_variable_index(var)] = 0.01
            def false_fn():
                pass
            tf.cond(self.scale[self._get_variable_index(var)] == 0.0, true_fn, false_fn)
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        
        grad_norm = neuron_norm(gradient)
        
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg_sq.assign(exp_avg_sq * self.beta + grad_norm * grad_norm * (1.0 - self.beta))

        bias_correction = 1.0 - self.beta ** self.step[self._get_variable_index(variable)]

        grad_normed = gradient / (tf.sqrt((exp_avg_sq / bias_correction)) + self.epsilon)
        grad_normed = nan_to_num(grad_normed, nan=0.0)

        variable.assign_add(grad_normed * -lr * self.scale[self._get_variable_index(variable)])

        if self.constraints and len(variable.shape) > 1:
            variable.assign_sub(neuron_mean(variable))
            variable.assign(variable / (neuron_norm(variable) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "epsilon": self.epsilon,
                "constraints": self.constraints,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass