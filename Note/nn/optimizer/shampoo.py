""" Shampoo
Implements Shampoo Optimizer Algorithm.
It has been proposed in `Shampoo: Preconditioned Stochastic Tensor
Optimization`__.
https://arxiv.org/abs/1802.09568

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def matrix_power(matrix, power):
    original_device = matrix.device if hasattr(matrix, 'device') else '/CPU:0'

    with tf.device('/CPU:0'):
        s, u, v = tf.linalg.svd(matrix)
        s_power = tf.pow(s, power)
        s_diag = tf.linalg.diag(s_power)
        result_cpu = tf.matmul(u, tf.matmul(s_diag, tf.transpose(v)))

    with tf.device(original_device):
        result = tf.identity(result_cpu)

    return result


class Shampoo(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-1,
        epsilon=1e-4,
        weight_decay=0.0,
        momentum=0.0,
        update_freq=1,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="shampoo",
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
        self.epsilon = epsilon
        self.momentum = momentum
        self.update_freq = update_freq

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.precond = []
        self.inv_precond = []
        self.momentum_buffer = []
        self.step = []
        for var in var_list:
            shape = var.shape.as_list()
            self.precond.append(dict())
            self.inv_precond.append(dict())
            for dim_id, dim in enumerate(shape):
                self.precond[-1]["precond_{}".format(dim_id)] = tf.Variable(self.epsilon * tf.eye(dim, dtype=var.dtype))
                self._track_variable(self.precond[-1]["precond_{}".format(dim_id)])
                self.inv_precond[-1]["inv_precond_{}".format(dim_id)] =  tf.Variable(tf.zeros((dim, dim), dtype=var.dtype))
                self._track_variable(self.inv_precond[-1]["inv_precond_{}".format(dim_id)])
            self.momentum_buffer.append(None)
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        order = len(gradient.shape)
        original_size = gradient.shape.as_list()
        
        if self.momentum_buffer[self._get_variable_index(variable)] == None:
            self.momentum_buffer[self._get_variable_index(variable)] = gradient
        
        if self.momentum > 0:
            gradient = gradient * (1 - self.momentum) + self.momentum_buffer[self._get_variable_index(variable)] * self.momentum

        if self.weight_decay > 0:
            gradient += variable * self.weight_decay

        # See Algorithm 2 for detail
        for dim_id, dim in enumerate(gradient.shape.as_list()):
            precond = self.precond[self._get_variable_index(variable)]["precond_{}".format(dim_id)]
            inv_precond = self.inv_precond[self._get_variable_index(variable)]["inv_precond_{}".format(dim_id)]

            # mat_{dim_id}(grad)
            current_rank = len(gradient.shape)
            perm = list(range(current_rank))
            perm[0], perm[dim_id] = perm[dim_id], perm[0]
            gradient = tf.transpose(gradient, perm=perm)
            transposed_size = gradient.shape.as_list()
            gradient = tf.reshape(gradient, (dim, -1))

            gradient_t = tf.transpose(gradient)
            precond.assign_add(tf.matmul(gradient, gradient_t))
            if self.step[self._get_variable_index(variable)] % self.update_freq == 0:
                inv_precond.assign(matrix_power(precond, -1.0 / order))

            if dim_id == order - 1:
                # finally
                gradient = tf.matmul(gradient_t, inv_precond)
                # grad: (-1, last_dim)
                gradient = tf.reshape(gradient, original_size)
            else:
                # if not final
                gradient = tf.matmul(inv_precond, gradient)
                # grad (dim, -1)
                gradient = tf.reshape(gradient, transposed_size)

        self.momentum_buffer[self._get_variable_index(variable)] = gradient
        variable.assign_add(gradient * -lr)
        
        self.step[self._get_variable_index(variable)] += 1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "update_freq": self.update_freq,
                "momentum_buffer": self.momentum_buffer,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass