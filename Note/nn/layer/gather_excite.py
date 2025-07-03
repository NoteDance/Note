""" Gather-Excite Attention Block

Paper: `Gather-Excite: Exploiting Feature Context in CNNs` - https://arxiv.org/abs/1810.12348

Official code here, but it's only partial impl in Caffe: https://github.com/hujie-frank/GENet

I've tried to support all of the extent both w/ and w/o params. I don't believe I've seen another
impl that covers all of the cases.

NOTE: extent=0 + extra_params=False is equivalent to Squeeze-and-Excitation

Hacked together by / Copyright 2024 NoteDance
"""
import math

import tensorflow as tf
from Note import nn


class GatherExcite(nn.Layer):
    """ Gather-Excite Attention Module
    """
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1./16, rd_channels=None,  rd_divisor=1, add_maxpool=False,
            act_layer=tf.nn.relu, norm_layer=nn.batch_norm, gate_layer=tf.nn.sigmoid):
        super().__init__()
        self.add_maxpool = add_maxpool
        act_layer = act_layer
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                padding = ((1 - 1) + 1 * (feat_size - 1)) // 2
                self.gather.add(
                            nn.conv2d(channels, input_size=channels, kernel_size=feat_size, strides=1, padding=padding, groups=channels))
                if norm_layer:
                    self.gather.add(nn.batch_norm(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    padding = ((2 - 1) + 1 * (3 - 1)) // 2
                    self.gather.add(
                        nn.conv2d(channels, input_size=channels, kernel_size=3, strides=2, padding=padding, groups=channels))
                    if norm_layer:
                        self.gather.add(nn.batch_norm(channels))
                    if i != num_conv - 1:
                        self.gather.add(act_layer)
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = nn.make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = nn.ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.identity()
        self.gate = gate_layer
        self.avg_pool2d = nn.avg_pool2d(kernel_size=self.gk, strides=self.gs, padding=self.gk // 2, count_include_pad=False)
        self.max_pool2d = nn.max_pool2d(kernel_size=self.gk, strides=self.gs, padding=self.gk // 2)

    def __call__(self, x):
        size = x.shape[1:3]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * tf.reduce_max(x, axis=(1, 2), keepdims=True)
            else:
                x_ge = self.avg_pool2d(x)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * self.max_pool2d(x)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[1] != 1 or x_ge.shape[2] != 1:
            x_ge = nn.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)