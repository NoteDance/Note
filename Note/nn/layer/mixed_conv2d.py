""" Note Mixed Convolution

Paper: MixConv: Mixed Depthwise Convolutional Kernels (https://arxiv.org/abs/1907.09595)

Hacked together by / Copyright 2025 NoteDance
"""

import tensorflow as tf
from Note import nn


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.Layer):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """
    def __init__(self, filters, kernel_size=3, input_size=None,
                 strides=1, padding='', dilations=1, depthwise=False, **kwargs):
        super().__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(input_size, num_groups)
        out_splits = _split_channels(filters, num_groups)
        self.input_size = sum(in_splits)
        self.filters = sum(out_splits)
        self.convs = []
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            if padding not in ('SAME', 'VALID'):
                padding = nn.get_padding(k, strides, dilations)
            conv_layer = nn.conv2d(out_ch, k, in_ch, strides=strides, padding=padding, dilations=dilations, groups=conv_groups, **kwargs)
            name = str(idx)
            setattr(self, name, conv_layer)
            self.convs.append(conv_layer)
        self.splits = in_splits

    def __call__(self, x):
        x_split = tf.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.convs)]
        x = tf.concat(x_out, 1)
        return x