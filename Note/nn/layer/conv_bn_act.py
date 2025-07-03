""" Conv2d + BN + Act

Hacked together by / Copyright 2025 NoteDance
"""
import tensorflow as tf
from Note import nn
from functools import partial


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(
        aa_layer,
        channels = None,
        stride = 2,
        enable = True,
        noop = nn.identity
):
    """ Anti-aliasing """
    if not aa_layer or not enable:
        return noop() if noop is not None else None

    if isinstance(aa_layer, str):
        aa_layer = aa_layer.lower().replace('_', '').replace('-', '')
        if aa_layer == 'avg' or aa_layer == 'avgpool':
            aa_layer = nn.avg_pool2d
        elif aa_layer == 'blur' or aa_layer == 'blurpool':
            aa_layer = nn.BlurPool2d
        elif aa_layer == 'blurpc':
            aa_layer = partial(nn.BlurPool2d, pad_mode='constant')

        else:
            assert False, f"Unknown anti-aliasing layer ({aa_layer})."

    try:
        return aa_layer(channels=channels, stride=stride)
    except TypeError as e:
        return aa_layer(stride)


class ConvNormAct(nn.Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            apply_norm: bool = True,
            apply_act: bool = True,
            norm_layer = nn.batch_norm,
            act_layer = tf.nn.relu,
            aa_layer = None,
            drop_layer = None,
            drop_rate = 0.,
    ):
        super().__init__()
        use_aa = aa_layer is not None and stride > 1
        padding = get_padding(kernel_size, stride=1 if aa_layer else stride, dilation=dilation)

        self.conv = nn.conv2d(
            out_channels,
            kernel_size,
            in_channels,
            strides=1 if use_aa else stride,
            padding=padding,
            dilations=dilation,
            groups=groups,
            use_bias=bias,
        )

        if apply_norm:
            self.bn = nn.Sequential()
            self.norm_layer = norm_layer(out_channels)
            self.act_layer = act_layer
            self.bn.add(self.norm_layer)
            if drop_layer:
                self.drop_layer = drop_layer(drop_rate)
                self.bn.add(self.drop_layer)
            self.bn.add(self.act_layer)
        else:
            self.bn = nn.Sequential()
            if drop_layer:
                self.drop_layer = drop_layer(drop_rate)
                self.bn.add(self.drop_layer)

        self.aa = create_aa(aa_layer, out_channels, stride=stride, enable=use_aa, noop=None)

    @property
    def in_channels(self):
        return self.conv.input_size

    @property
    def out_channels(self):
        return self.conv.output_size

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        aa = self.aa
        if aa is not None:
            x = self.aa(x)
        return x