""" Bilinear-Attention-Transform and Non-Local Attention

Paper: `Non-Local Neural Networks With Grouped Bilinear Attentional Transforms`
    - https://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html
Adapted from original code: https://github.com/BA-Transform/BAT-Image-Classification

Copyright 2025 NoteDance
"""
import tensorflow as tf
from Note import nn


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class NonLocalAttn(nn.Layer):
    """Spatial NL block for image classification.

    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(self, in_channels, use_scale=True,  rd_ratio=1/8, rd_channels=None, rd_divisor=8, **kwargs):
        super().__init__()
        nn.Model.add()
        if rd_channels is None:
            rd_channels = nn.make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels ** -0.5 if use_scale else 1.0
        self.t = nn.conv2d(rd_channels, 1, in_channels, strides=1, use_bias=True)
        self.p = nn.conv2d(rd_channels, 1, in_channels, strides=1, use_bias=True)
        self.g = nn.conv2d(rd_channels, 1, in_channels, strides=1, use_bias=True)
        self.z = nn.conv2d(in_channels, 1, rd_channels, strides=1, use_bias=True)
        self.norm = nn.batch_norm(in_channels)
        nn.Model.apply(self.init_weights)

    def __call__(self, x):
        shortcut = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        B, H, W, C = t.shape
        t = tf.reshape(t, [B, -1, C])
        p = tf.reshape(p, [B, -1, C])
        g = tf.reshape(g, [B, -1, C])

        att = tf.matmul(t, p, transpose_b=True) * self.scale
        att = tf.nn.softmax(att, axis=-1)
        x = tf.matmul(att, g)

        x = tf.reshape(x, [B, H, W, -1])
        x = self.z(x)
        x = self.norm(x) + shortcut

        return x
    
    def init_weights(self, l):
        if isinstance(l, nn.conv2d):
            l.weight.assign(nn.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu'))
        elif isinstance(l, nn.batch_norm):
            l.gamma.assign(nn.constant_(l.gamma, 0))
        elif isinstance(l, nn.group_norm):
            l.gamma.assign(nn.constant_(l.gamma, 0))


class BilinearAttnTransform(nn.Layer):

    def __init__(self, in_channels, block_size, groups, act_layer=tf.nn.relu, norm_layer=nn.batch_norm):
        super().__init__()
        padding = get_padding(kernel_size=1, stride=1, dilation=1)
        self.conv1 = nn.conv2d(groups, 1, in_channels, strides=1, padding=padding, groups=1, dilations=1)
        self.norm_layer1 = norm_layer(groups)
        self.act_layer1 = act_layer
        self.conv_p = nn.conv2d(block_size * block_size * groups, kernel_size=(block_size, 1), input_size=groups)
        self.conv_q = nn.conv2d(block_size * block_size * groups, kernel_size=(1, block_size), input_size=groups)
        self.conv2 = nn.conv2d(in_channels, 1, in_channels, strides=1, padding=padding, groups=1, dilations=1)
        self.norm_layer2 = norm_layer(in_channels)
        self.act_layer2 = act_layer
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels
    
    def resize_mat(x, t: int):
        B, block_size, block_size1, C = x.shape
        assert block_size == block_size1
        if t <= 1:
            return x
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = tf.reshape(x, [B * C, block_size, block_size])
        eye_t = tf.eye(t, dtype=x.dtype)
        x = tf.expand_dims(x, axis=-1)
        x = tf.expand_dims(x, axis=-1)
        x = x * eye_t
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4])
        x = tf.reshape(x, [B * C, block_size * t, block_size * t])
        x = tf.reshape(x, [B, C, block_size * t, block_size * t])
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x

    def __call__(self, x):
        assert x.shape[1] % self.block_size == 0
        assert x.shape[2] % self.block_size == 0
        B, H, W, C = x.shape
        out = self.conv1(x)
        out = self.norm_layer1(out)
        out = self.act_layer1(out)
        rp = nn.adaptive_max_pool2d(out, (self.block_size, 1))
        cp = nn.adaptive_max_pool2d(out, (1, self.block_size))
        p = tf.nn.sigmoid(tf.reshape(self.conv_p(rp), (B, self.block_size, self.block_size, self.groups)))
        q = tf.nn.sigmoid(tf.reshape(self.conv_q(cp), (B, self.block_size, self.block_size, self.groups)))
        p = p / tf.reduce_sum(p, axis=2, keepdims=True)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        p = tf.reshape(p, [B, self.block_size, self.block_size, self.groups, 1])
        p = tf.tile(p, [x.shape[0], self.block_size, self.block_size, self.groups, C // self.groups])
        p = tf.reshape(p, (B, self.block_size, self.block_size, C))
        q = tf.reshape(q, [B, self.block_size, self.block_size, self.groups, 1])
        q = tf.tile(q, [x.shape[0], self.block_size, self.block_size, self.groups, C // self.groups])
        q = tf.reshape(q, (B, self.block_size, self.block_size, C))
        p = self.resize_mat(p, H // self.block_size)
        q = self.resize_mat(q, W // self.block_size)
        y = tf.matmul(p, x)
        y = tf.matmul(y, q)

        y = self.conv2(y)
        y = self.conv2(y)
        y = self.norm_layer2(y)
        y = self.act_layer2(y)
        return y


class BatNonLocalAttn(nn.Layer):
    """ BAT
    Adapted from: https://github.com/BA-Transform/BAT-Image-Classification
    """

    def __init__(
            self, in_channels, block_size=7, groups=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
            drop_rate=0.2, act_layer=tf.nn.relu, norm_layer=nn.batch_norm, **_):
        super().__init__()
        if rd_channels is None:
            rd_channels = nn.make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        padding = get_padding(kernel_size=1, stride=1, dilation=1)
        self.conv1 = nn.conv2d(rd_channels, 1, in_channels, strides=1, padding=padding, groups=1, dilations=1)
        self.norm_layer1 = norm_layer(rd_channels)
        self.act_layer1 = act_layer
        self.ba = BilinearAttnTransform(rd_channels, block_size, groups, act_layer=act_layer, norm_layer=norm_layer)
        self.conv2 = nn.conv2d(in_channels, 1, rd_channels, strides=1, padding=padding, groups=1, dilations=1)
        self.norm_layer1 = norm_layer(rd_channels)
        self.act_layer1 = act_layer
        self.dropout = nn.dropout(drop_rate)

    def __call__(self, x):
        xl = self.conv1(x)
        xl = self.norm_layer1(xl)
        xl = self.act_layer1(xl)
        y = self.ba(xl)
        y = self.conv2(y)
        y = self.norm_layer2(y)
        y = self.act_layer2(y)
        y = self.dropout(y)
        return y + x