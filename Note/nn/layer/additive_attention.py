import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.Module import Module


class additive_attention:
    def __init__(self,input_size=None, use_scale=True, dtype='float32'):
        self.use_scale = use_scale
        self.dtype=dtype
        if input_size!=None and use_scale:
            self.scale = initializer([input_size], 'Xavier', dtype)
            self.param=[self.scale]
            Module.param.extend(self.param)
    
    def build(self):
        self.output_size=self.input_size
        if self.input_size!=None and self.use_scale:
            self.scale = initializer([self.input_size], 'Xavier', self.dtype)
            self.param=[self.scale]
            Module.param.extend(self.param)
        return

    def __call__(self, query, key):
        """Calculates attention scores as a nonlinear sum of query and key.

        Args:
            query: Query tensor of shape `[batch_size, Tq, dim]`.
            key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
            Tensor of shape `[batch_size, Tq, Tv]`.
        """
        if query.dtype!=self.dtype:
            query=tf.cast(query,self.dtype)
        if key.dtype!=self.dtype:
            key=tf.cast(key,self.dtype)
        if self.input_size==None:
            self.input_size=query.shape[-1]
            self.build()
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