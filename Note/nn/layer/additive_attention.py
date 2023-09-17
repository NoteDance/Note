import tensorflow as tf
from Note.nn.initializer import initializer


class additive_attention:
    def __init__(self,input_size, use_scale=True, dtype='float32'):
        self.use_scale = use_scale
        self.param=[]
        if use_scale:
            self.scale = initializer([input_size], 'Xavier', dtype)
            self.param.append(self.scale)


    def output(self, query, key):
        """Calculates attention scores as a nonlinear sum of query and key.

        Args:
            query: Query tensor of shape `[batch_size, Tq, dim]`.
            key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
            Tensor of shape `[batch_size, Tq, Tv]`.
        """
        # Reshape tensors to enable broadcasting.
        # Reshape into [batch_size, Tq, 1, dim].
        q_reshaped = tf.expand_dims(query, axis=-2)
        # Reshape into [batch_size, 1, Tv, dim].
        k_reshaped = tf.expand_dims(key, axis=-3)
        if self.use_scale:
            scale = self.scale
        else:
            scale = 1.0
        return tf.reduce_sum(scale * tf.tanh(q_reshaped + k_reshaped), axis=-1)