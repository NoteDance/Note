import numbers
import tensorflow as tf
from Note import nn


class AdaNorm(nn.Layer):
    def __init__(self, normalized_shape, k: float = 0.1, eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.k = k
        self.eps = eps
        self.weight = nn.Parameter(tf.ones(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(tf.zeros(self.normalized_shape))
        else:
            self.bias = None

    def __call__(self, input):
        mean = tf.reduce_mean(input, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.pow(input - mean, 2), axis=-1, keepdims=True) + self.eps
    
        input_norm = (input - mean) * tf.math.rsqrt(var)
        
        adanorm = self.weight * (1 - self.k * input_norm) * input_norm

        if self.bias is not None:
            adanorm = adanorm + self.bias
    
        return adanorm