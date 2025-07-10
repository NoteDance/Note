import tensorflow as tf
from Note import nn


class RMSNorm(nn.Layer):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(tf.ones(d))

        if self.bias:
            self.offset = nn.Parameter(tf.zeros(d))

    def __call__(self, x):
        if x.dtype != tf.float32:
            x=tf.cast(x, tf.float32)
        if self.p < 0. or self.p > 1.:
            norm_x = tf.norm(x, axis=-1, keepdims=True)
            d_x = self.d
        else:
            partial_size = tf.cast(self.d * self.p, tf.int32)
            partial_x = x[..., :partial_size]

            norm_x = tf.norm(partial_x, axis=-1, keepdims=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
