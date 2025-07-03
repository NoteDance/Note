""" Global Context Attention Block

Paper: `GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond`
    - https://arxiv.org/abs/1904.11492

Official code consulted as reference: https://github.com/xvjiarui/GCNet

Hacked together by / Copyright 2024 NoteDance
"""
import tensorflow as tf
from Note import nn


class GlobalContext(nn.Layer):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1./8, rd_channels=None, rd_divisor=1, act_layer=tf.nn.relu, gate_layer=tf.nn.sigmoid):
        super().__init__()
        self.conv_attn = nn.conv2d(1, kernel_size=1, input_size=channels, use_bias=True) if use_attn else None

        if rd_channels is None:
            rd_channels = nn.make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = nn.ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=nn.layer_norm)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = nn.ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=nn.layer_norm)
        else:
            self.mlp_scale = None

        self.gate = gate_layer
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            self.mlp_add.fc2.weight.assign(tf.zeros(self.mlp_add.fc2.weight.shape))

    def __call__(self, x):
        B, H, W, C = x.shape

        if self.conv_attn is not None:
            attn = tf.reshape(self.conv_attn(x), (B, 1, H * W))  # (B, 1, H * W)
            attn = tf.expand_dims(tf.nn.softmax(attn, axis=-1), axis=3)  # (B, 1, H * W, 1)
            context = tf.matmul(tf.expand_dims(tf.reshape(tf.transpose(x, (0, 3, 1, 2)), (B, C, H * W)), axis=1), attn)
            context = tf.reshape(context, (B, 1, 1, C))
        else:
            context = tf.reduce_mean(x, axis=(1, 2), keepdims=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x