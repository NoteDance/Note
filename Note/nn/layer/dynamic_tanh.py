import tensorflow as tf
from Note import nn


class DynamicTanh:
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(tf.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(tf.ones(normalized_shape))
        self.bias = nn.Parameter(tf.zeros(normalized_shape))

    def __call__(self, x):
        x = tf.nn.tanh(self.alpha * x)
        x = x * self.weight + self.bias
        return x