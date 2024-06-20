""" Selective Kernel Convolution/Attention

Paper: Selective Kernel Networks (https://arxiv.org/abs/1903.06586)

Hacked together by / Copyright 2024 NoteDance
"""
import tensorflow as tf
from Note import nn
from Note.nn.layer import batch_norm,avg_pool2d,spatial_dropout2d


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class SelectiveKernelAttn:
    def __init__(self, channels, num_paths=2, attn_channels=32, act_layer=tf.nn.relu, norm_layer=nn.batch_norm):
        """ Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        self.num_paths = num_paths
        self.fc_reduce = nn.conv2d(attn_channels, 1, channels, use_bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer
        self.fc_select = nn.conv2d(channels * num_paths, 1, attn_channels, use_bias=False)

    def __call__(self, x):
        x = tf.reduce_mean(tf.reduce_sum(x, axis=-1), (1, 2), keepdims=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, (B, H, W, self.num_paths, C // self.num_paths))
        x = tf.nn.softmax(x, axis=3)
        return x


class SelectiveKernel:

    def __init__(self, in_channels, out_channels=None, kernel_size=None, stride=1, dilation=1, groups=1,
                 rd_ratio=1./16, rd_channels=None, rd_divisor=8, keep_3x3=True, split_input=True,
                 act_layer=tf.nn.relu, norm_layer=batch_norm, aa_layer=avg_pool2d, drop_layer=spatial_dropout2d, drop_rate=0.):
        """ Selective Kernel Convolution Module

        As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.

        Largest change is the input split, which divides the input channels across each convolution path, this can
        be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
        the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
        a noteworthy increase in performance over similar param count models without this attention layer. -Ross W
        
        """
        out_channels = out_channels or in_channels
        kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch. 5x5 -> 3x3 + dilation
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)

        self.paths = nn.Sequential()
        for k, d in zip(kernel_size, dilation):
            padding = get_padding(kernel_size, stride=1 if aa_layer else stride, dilation=dilation)
            self.paths.add(nn.conv2d(out_channels, k, in_channels, strides=1 if aa_layer else stride, padding=padding, groups=groups, dilation=d))
            self.paths.add(norm_layer(out_channels))
            self.paths.add(drop_layer(drop_rate))
            self.paths.add(act_layer)
            if aa_layer is not None:
                self.paths.add(aa_layer(stride, stride))

        attn_channels = rd_channels or nn.make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def __call__(self, x):
        if self.split_input:
            x_split = tf.split(x, self.in_channels // self.num_paths, -1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]
        x = tf.stack(x_paths, axis=-1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = tf.reduce_sum(x, dim=-1)
        return x
