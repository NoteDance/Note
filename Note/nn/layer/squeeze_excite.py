""" Squeeze-and-Excitation Channel Attention

An SE implementation originally based on PyTorch SE-Net impl.
Has since evolved with additional functionality / configuration.

Paper: `Squeeze-and-Excitation Networks` - https://arxiv.org/abs/1709.01507

Also included is Effective Squeeze-Excitation (ESE).
Paper: `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667

Hacked together by / Copyright 2024 NoteDance
"""
import tensorflow as tf
from Note import nn
from Note.nn.activation import activation_dict


class SEModule:
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            bias=True, act_layer=tf.nn.relu, norm_layer=None, gate_layer=tf.nn.sigmoid):
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = nn.make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.conv2d(rd_channels, 1, channels, use_bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.identity()
        self.act = act_layer
        self.fc2 = nn.conv2d(channels, 1, rd_channels, use_bias=bias)
        self.gate = gate_layer

    def __call__(self, x):
        x_se = tf.reduce_mean(x, axis=(2, 3), keepdims=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * tf.reduce_max(x, axis=(2, 3), keepdims=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


SqueezeExcite = SEModule  # alias


class EffectiveSEModule:
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """
    def __init__(self, channels, add_maxpool=False, gate_layer=activation_dict['hard_sigmoid'], **_):
        self.add_maxpool = add_maxpool
        self.zeropadding2d = nn.zeropadding2d(padding=0)
        self.fc = nn.conv2d(channels, 1, channels)
        self.gate = gate_layer

    def __call__(self, x):
        x_se = tf.reduce_mean(x, axis=(2, 3), keepdims=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * tf.reduce_max(x, axis=(2, 3), keepdims=True)
        x_se = self.zeropadding2d(x_se)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


EffectiveSqueezeExcite = EffectiveSEModule  # alias


class SqueezeExciteCl:
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8,
            bias=True, act_layer=tf.nn.relu, gate_layer=tf.nn.sigmoid):
        if not rd_channels:
            rd_channels = nn.make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.dense(rd_channels, channels, use_bias=bias)
        self.act = act_layer
        self.fc2 = nn.dense(channels, rd_channels, use_bias=bias)
        self.gate = gate_layer

    def __call__(self, x):
        x_se = tf.reduce_mean(x, axis=(1, 2), keepdims=True)  # FIXME avg dim [1:n-1], don't assume 2D NHWC
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)