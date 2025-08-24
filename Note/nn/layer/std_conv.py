""" Convolution with Weight Standardization (StdConv and ScaledStdConv)

StdConv:
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
Code: https://github.com/joe-siyuan-qiao/WeightStandardization

ScaledStdConv:
Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692
Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Hacked together by / copyright NoteDance, 2025.
"""
import tensorflow as tf
from Note import nn


class StdConv2d(nn.conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, strides=1, padding=None,
            dilations=1, groups=1, use_bias=False, eps=1e-6):
        if padding is None:
            padding = nn.get_padding(kernel_size, strides, dilations)
        super().__init__(
            out_channels, kernel_size, in_channel, strides=strides,
            padding=padding, dilations=dilations, groups=groups, use_bias=use_bias)
        self.eps = eps

    def __call__(self, x):
        weight = tf.reshape(self.weight, (1, -1, self.output_size))
        mean, var = tf.nn.moments(weight, axes=[0, 1], keepdims=True)
        weight = tf.nn.batch_normalization(
                weight,
                mean=mean,
                variance=var,
                offset=None,
                scale=None,
                variance_epsilon=self.eps
            )
        self.weight.assign(tf.reshape(weight, self.weight.shape))
        x = nn.conv2d_func(x, self.weight, self.bias, self.strides, self.padding, self.dilations, self.groups)
        return x


class StdConv2dSame(nn.conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, strides=1, padding='SAME',
            dilations=1, groups=1, use_bias=False, eps=1e-6):
        if padding not in ('SAME', 'VALID'):
            padding = nn.get_padding(kernel_size, strides, dilations)
        super().__init__(
            out_channels, kernel_size, in_channel, strides=strides, padding=padding, dilations=dilations,
            groups=groups, use_bias=use_bias)
        self.eps = eps

    def __call__(self, x):
        weight = tf.reshape(self.weight, (1, -1, self.output_size))
        mean, var = tf.nn.moments(weight, axes=[0, 1], keepdims=True)
        weight = tf.nn.batch_normalization(
                weight,
                mean=mean,
                variance=var,
                offset=None,
                scale=None,
                variance_epsilon=self.eps
            )
        self.weight.assign(tf.reshape(weight, self.weight.shape))
        x = nn.conv2d_func(x, self.weight, self.bias, self.strides, self.padding, self.dilations, self.groups)
        return x


class ScaledStdConv2d(nn.conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, strides=1, padding=None,
            dilations=1, groups=1, use_bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        if padding is None:
            padding = nn.get_padding(kernel_size, strides, dilations)
        super().__init__(
            out_channels, kernel_size, in_channels, strides=strides, padding=padding, dilations=dilations,
            groups=groups, use_bias=use_bias)
        self.gain = nn.Parameter(tf.fill((1, 1, 1, self.output_size), gain_init))
        self.scale = gamma * tf.size(self.weight[0]) ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def __call__(self, x):
        weight = tf.reshape(self.weight, (1, -1, self.output_size))
        mean, var = tf.nn.moments(weight, axes=[0, 1], keepdims=True)
        gamma = tf.cast(self.gain * self.scale, weight.dtype)
        gamma = tf.reshape(gamma, [self.output_size])
        weight = tf.nn.batch_normalization(
                weight,
                mean=mean,
                variance=var,
                offset=None,
                scale=gamma,
                variance_epsilon=self.eps
            )
        self.weight.assign(tf.reshape(weight, self.weight.shape))
        return nn.conv2d_func(x, self.weight, self.bias, self.strides, self.padding, self.dilations, self.groups)


class ScaledStdConv2dSame(nn.conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, strides=1, padding='SAME',
            dilations=1, groups=1, use_bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        if padding not in ('SAME', 'VALID'):
            padding = nn.get_padding(kernel_size, strides, dilations)
        super().__init__(
            out_channels, kernel_size, in_channels, strides=strides, padding=padding, dilations=dilations,
            groups=groups, use_bias=use_bias)
        self.gain = nn.Parameter(tf.fill((1, 1, 1, self.out_channels), gain_init))
        self.scale = gamma * tf.size(self.weight[0]) ** -0.5
        self.eps = eps

    def __call__(self, x):
        weight = tf.reshape(self.weight, (1, -1, self.output_size))
        mean, var = tf.nn.moments(weight, axes=[0, 1], keepdims=True)
        gamma = tf.cast(self.gain * self.scale, weight.dtype)
        gamma = tf.reshape(gamma, [self.output_size])
        weight = tf.nn.batch_normalization(
                weight,
                mean=mean,
                variance=var,
                offset=None,
                scale=gamma,
                variance_epsilon=self.eps
            )
        self.weight.assign(tf.reshape(weight, self.weight.shape))
        return nn.conv2d_func(x, self.weight, self.bias, self.strides, self.padding, self.dilations, self.groups)