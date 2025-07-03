import tensorflow as tf
from Note import nn
from Note.nn.initializer import initializer


class RMSNorm(nn.Layer):
    def __init__(self, dims: int, eps: float = 1e-6, dtype='float32'):
        super().__init__()
        self.gamma = initializer((dims,), 'ones', dtype)
        self.eps = eps

    def __call__(self, x):
        n = tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True) + self.eps)
        return self.gamma * x * n