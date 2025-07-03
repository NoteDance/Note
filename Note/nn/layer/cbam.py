""" CBAM (sort-of) Attention

Experimental impl of CBAM: Convolutional Block Attention Module: https://arxiv.org/abs/1807.06521

WARNING: Results with these attention layers have been mixed. They can significantly reduce performance on
some tasks, especially fine-grained it seems. I may end up removing this impl.

Hacked together by / Copyright 2024 NoteDance
"""
import tensorflow as tf
from Note import nn


class ChannelAttn(nn.Layer):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """
    def __init__(
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1,
            act_layer=tf.nn.relu, gate_layer=tf.nn.sigmoid, mlp_bias=False):
        super().__init__()
        if not rd_channels:
            rd_channels = nn.make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.conv2d(rd_channels, 1, channels, use_bias=mlp_bias)
        self.act = act_layer
        self.fc2 = nn.conv2d(channels, 1, rd_channels, use_bias=mlp_bias)
        self.gate = gate_layer

    def __call__(self, x):
        x_avg = self.fc2(self.act(self.fc1(tf.reduce_mean(x, (1, 2), keepdims=True))))
        x_max = self.fc2(self.act(self.fc1(tf.reduce_max(x, (1, 2), keepdims=True))))
        return x * self.gate(x_avg + x_max)


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """
    def __init__(
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1,
            act_layer=tf.nn.relu, gate_layer=tf.nn.sigmoid, mlp_bias=False):
        super(LightChannelAttn, self).__init__(
            channels, rd_ratio, rd_channels, rd_divisor, act_layer, gate_layer, mlp_bias)

    def __call__(self, x):
        x_pool = 0.5 * tf.reduce_mean(x, (1, 2), keepdims=True) + 0.5 * tf.reduce_max(x, (1, 2), keepdims=True)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * tf.nn.sigmoid(x_attn)


class SpatialAttn(nn.Layer):
    """ Original CBAM spatial attention module
    """
    def __init__(self, kernel_size=7, gate_layer=tf.nn.sigmoid):
        super().__init__()
        self.conv = nn.conv2d(1, kernel_size, 2)
        self.norm_layer = nn.batch_norm(2)
        self.act_layer = tf.nn.relu
        self.gate = gate_layer

    def __call__(self, x):
        x_attn = tf.concat([tf.reduce_mean(x, axis=-1, keepdims=True), tf.reduce_max(x, axis=-1, keepdims=True)], axis=-1)
        x_attn = self.conv(x_attn)
        x_attn = self.norm_layer(x_attn)
        x_attn = self.act_layer(x_attn)
        return x * self.gate(x_attn)


class LightSpatialAttn(nn.Layer):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """
    def __init__(self, kernel_size=7, gate_layer=tf.nn.sigmoid):
        super().__init__()
        self.conv = nn.conv2d(1, kernel_size, 1)
        self.norm_layer = nn.batch_norm(1)
        self.act_layer = tf.nn.relu
        self.gate = gate_layer

    def __call__(self, x):
        x_attn = 0.5 * tf.reduce_mean(x, axis=-1, keepdims=True) + 0.5 * tf.reduce_max(x, axis=-1, keepdims=True)
        x_attn = self.conv(x_attn)
        x_attn = self.norm_layer(x_attn)
        x_attn = self.act_layer(x_attn)
        return x * self.gate(x_attn)


class CbamModule(nn.Layer):
    def __init__(
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1,
            spatial_kernel_size=7, act_layer=tf.nn.relu, gate_layer=tf.nn.sigmoid, mlp_bias=False):
        super().__init__()
        self.channel = ChannelAttn(
            channels, rd_ratio=rd_ratio, rd_channels=rd_channels,
            rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        self.spatial = SpatialAttn(spatial_kernel_size, gate_layer=gate_layer)

    def __call__(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightCbamModule(nn.Layer):
    def __init__(
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1,
            spatial_kernel_size=7, act_layer=tf.nn.relu, gate_layer=tf.nn.sigmoid, mlp_bias=False):
        super().__init__()
        self.channel = LightChannelAttn(
            channels, rd_ratio=rd_ratio, rd_channels=rd_channels,
            rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def __call__(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x