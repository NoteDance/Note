""" Global Response Normalization Module

Based on the GRN layer presented in
`ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders` - https://arxiv.org/abs/2301.00808

This implementation
* works for both NCHW and NHWC tensor layouts
* uses affine param names matching existing torch norm layers
* slightly improves eager mode performance via fused addcmul

Hacked together by / Copyright 2024 NoteDance
"""

import tensorflow as tf
from Note import nn


class GlobalResponseNorm(nn.Layer):
    """ Global Response Normalization layer
    """
    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = nn.Parameter(tf.zeros(dim))
        self.bias = nn.Parameter(tf.zeros(dim))

    def __call__(self, x):
        x_g = tf.norm(x, ord=2, axis=self.spatial_dim, keepdims=True)
        x_n = x_g / (tf.reduce_mean(x_g, axis=self.channel_dim, keepdims=True) + self.eps)
        bias_reshaped = tf.reshape(self.bias, self.wb_shape)
        weight_reshaped = tf.reshape(self.weight, self.wb_shape)
        product = tf.multiply(x, x_n)
        weighted_product = tf.multiply(weight_reshaped, product)
        return x + tf.add(bias_reshaped, weighted_product)