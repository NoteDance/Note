""" SM3
https://arxiv.org/abs/1901.11150

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def reduce_max_except_dim(x, dim):
    r"""Perform reduce-max along all dimensions except the given dim.

    :param x: tf.Tensor. tensor to reduce-max.
    :param dim: int. dimension to exclude.
    """
    rank = len(x.shape)
    if rank == 0:
        return x

    if dim >= rank:
        raise ValueError(f'[-] given dim is bigger than rank. {dim} >= {rank}')

    for d in range(rank):
        if d != dim:
            x = tf.reduce_max(x, axis=d, keepdims=True)
    return x


class SM3(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-1,
        epsilon=1e-30,
        momentum=0.0,
        beta=0.0,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sm3",
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
        self.epsilon = epsilon
        self.momentum = momentum
        self.beta = beta
    
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
            shape = var.shape
            rank = len(shape)
            self.momentum_buffer[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum_buffer"
                                                    )
            if tf.keras.backend.is_sparse(var):
                self.accumulator_0[self._get_variable_index(var)] = tf.Variable(tf.zeros_like(shape[0]))
                self._track_variable(self.accumulator_0[self._get_variable_index(var)])
            elif rank == 0:
                self.accumulator_0[self._get_variable_index(var)] = tf.Variable(tf.zeros_like(var))
                self._track_variable(self.accumulator_0[self._get_variable_index(var)])
            else:
                for i in range(rank):
                    self.accumulator[self._get_variable_index(var)][f'accumulator_{i}'] = tf.Variable(tf.zeros(
                        [1] * i + [shape[i]] + [1] * (rank - 1 - i), dtype=var.dtype
                            ))
                    self._track_variable(self.accumulator[self._get_variable_index(var)][f'accumulator_{i}'])

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.accumulator_0 = []
        self.accumulator = []
        for var in var_list:
            shape = var.shape
            rank = len(shape)
            self.momentum_buffer.append(self.add_variable_from_reference(
                                reference_variable=var, name="momentum_buffer"
                                                    ))
            if tf.keras.backend.is_sparse(var):
                self.accumulator_0.append(tf.Variable(tf.zeros_like(shape[0])))
                self._track_variable(self.accumulator_0[-1])
            elif rank == 0:
                self.accumulator_0.append(tf.Variable(tf.zeros_like(var)))
                self._track_variable(self.accumulator_0[-1])
            else:
                self.accumulator.append(dict())
                for i in range(rank):
                    self.accumulator[-1][f'accumulator_{i}'] = tf.Variable(tf.zeros(
                        [1] * i + [shape[i]] + [1] * (rank - 1 - i), dtype=var.dtype
                            ))
                    self._track_variable(self.accumulator[-1][f'accumulator_{i}'])
    
    def make_sparse(self, grad, values):
        if tf.equal(tf.size(grad.indices), 0) or tf.equal(tf.size(values), 0):
            empty_indices = tf.zeros((0, tf.shape(grad.dense_shape)[0]), dtype=tf.int64)
            empty_values = tf.zeros((0,), dtype=values.dtype)
            return tf.SparseTensor(
                indices=empty_indices,
                values=empty_values,
                dense_shape=grad.dense_shape
            )
    
        return tf.SparseTensor(
            indices=grad.indices,
            values=values,
            dense_shape=grad.dense_shape
        )
    
    def coalesce_sparse(self, sp: tf.SparseTensor) -> tf.SparseTensor:
        dense_shape = tf.cast(sp.dense_shape, tf.int64)
        multipliers = tf.concat([
            tf.math.cumprod(dense_shape[1:], reverse=False),
            tf.constant([1], dtype=tf.int64)
        ], axis=0)
        linear_idx = tf.reduce_sum(sp.indices * multipliers, axis=1)
        unique_idx, segment_ids = tf.unique(linear_idx)
        summed_vals = tf.math.unsorted_segment_sum(
            sp.values, segment_ids, tf.shape(unique_idx)[0]
        )
        unraveled = tf.unravel_index(unique_idx, sp.dense_shape)
        new_indices = tf.stack(unraveled, axis=1)
        return tf.SparseTensor(new_indices, summed_vals, sp.dense_shape)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step += 1
        
        shape = gradient.shape
        rank = len(shape)
        
        if tf.keras.backend.is_sparse(gradient):
            grad_coalesced = self.coalesce_sparse(gradient)
        
            acc = self.accumulator_0[self._get_variable_index(variable)]
            
            idx0 = grad_coalesced.indices[:, 0]
            update_values = tf.gather(acc, idx0, axis=0)
            if self.beta > 0.0:
                update_values = update_values * self.beta
            sq = grad_coalesced.values * grad_coalesced.values
            update_values = update_values + (1.0 - self.beta) * sq
            
            sp_updated = self.make_sparse(grad_coalesced, update_values)
            dense_updated = tf.sparse.to_dense(sp_updated)
            axes = list(range(1, tf.rank(dense_updated)))
            nu_max = tf.reduce_max(dense_updated, axis=axes)
            
            if self.beta > 0.0:
                acc.assign(tf.maximum(acc, tf.expand_dims(nu_max, axis=list(range(1, tf.rank(acc))))))
            else:
                acc.assign(tf.expand_dims(nu_max, axis=list(range(1, tf.rank(acc)))))
            
            update_values = update_values + self.epsilon
            update_values = tf.math.rsqrt(update_values)
            update_values = update_values * grad_coalesced.values

            update = self.make_sparse(grad_coalesced, update_values)
        else:
            update = self.accumulator_0[self._get_variable_index(variable)]
            for i in range(1, rank):
                update = tf.minimum(update, self.accumulator[self._get_variable_index(variable)][f'accumulator_{i}'])

            if self.beta > 0.0:
                update = update * self.beta
            update = update + gradient * gradient * (1.0 - self.beta)

            for i in range(rank):
                acc = self.accumulator[self._get_variable_index(variable)][f'accumulator_{i}']
                nu_max = reduce_max_except_dim(update, i)
                if self.beta > 0.0:
                    acc.assign(tf.maximum(acc, nu_max))
                else:
                    acc.assign(nu_max)

            update = tf.math.rsqrt(update + self.epsilon) * gradient

            if self.momentum > 0.0:
                m = self.momentum_buffer[self._get_variable_index(variable)]
                m.assign(m * self.momentum + update * (1.0 - self.momentum))
                update = m

        variable.assign_add(update * -lr)          

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "beta": self.beta,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass