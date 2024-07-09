import tensorflow as tf
from Note.nn.initializer import initializer


class RMSNorm:
    def __init__(self, dims: int, eps: float = 1e-6, dtype='float32'):
        self.gamma = initializer((dims,), 'ones', dtype)
        self.eps = eps
        self.param = [self.gamma]

    def __call__(self, x):
        n = tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True) + self.eps)
        return self.gamma * x * n