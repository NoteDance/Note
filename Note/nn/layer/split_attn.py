""" Split Attention Conv2d (for ResNeSt Models)

Paper: `ResNeSt: Split-Attention Networks` - /https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl at https://github.com/zhanghang1989/ResNeSt

Hacked together by / Copyright 2024 NoteDance
"""
import tensorflow as tf
from Note import nn


class RadixSoftmax:
    def __init__(self, radix, cardinality):
        self.radix = radix
        self.cardinality = cardinality

    def __call__(self, x):
        batch = x.shape[0]
        if self.radix > 1:
            x = tf.transpose(tf.reshape(x, (batch, self.cardinality, self.radix, -1), (0, 1, 3, 2)))
            x = tf.nn.softmax(x, axis=1)
            x = tf.reshape(x, (batch, -1))
        else:
            x = tf.nn.sigmoid(x)
        return x


class SplitAttn(nn.Layer):
    """Split-Attention (aka Splat)
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=tf.nn.relu, norm_layer=None, drop_layer=None, **kwargs):
        super().__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = nn.make_divisible(in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.conv2d(
            mid_chs, kernel_size, in_channels, stride, padding, dilations=dilation,
            groups=groups * radix, use_bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.identity()
        self.drop = drop_layer() if drop_layer is not None else nn.identity()
        self.act0 = act_layer
        self.fc1 = nn.conv2d(attn_chs, 1, out_channels, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.identity()
        self.act1 = act_layer
        self.fc2 = nn.conv2d(mid_chs, 1, attn_chs, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)

        B, H, W, RC = x.shape
        if self.radix > 1:
            x = tf.reshape(x, (B, H, W, self.radix, RC // self.radix))
            x_gap = tf.reduce_sum(x, axis=-1)
        else:
            x_gap = x
        x_gap = tf.reduce_mean(x_gap, (1, 2), keepdims=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = tf.reshape(self.rsoftmax(x_attn), (B, 1, 1, -1))
        if self.radix > 1:
            out = tf.reduce_sum((x * tf.reshape(x_attn, (B, 1, 1, self.radix, RC // self.radix))), axis=-1)
        else:
            out = x * x_attn
        return out