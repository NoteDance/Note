"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

Hacked together by NoteDance
"""
from functools import partial
from typing import Optional

import tensorflow as tf
from Note import nn
from Note.nn.layer import identity
import numpy as np


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d:
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(
            self,
            channels: Optional[int] = None,
            filt_size: int = 3,
            stride: int = 2,
            pad_mode: str = 'REFLECT',
    ) -> None:
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.pad_mode = pad_mode
        pad = get_padding(filt_size, stride, dilation=1)
        self.padding = [[0, 0], [pad, pad], [pad, pad], [0, 0]]

        coeffs = tf.convert_to_tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(np.float32))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :]
        blur_filter = tf.transpose(blur_filter, (2, 3, 0, 1))
        if channels is not None:
            blur_filter = tf.tile(blur_filter, (1, 1, 1, self.channels))
        self.filt = blur_filter

    def __call__(self, x):
        x = tf.pad(x, self.padding, mode=self.pad_mode)
        if self.channels is None:
            channels = x.shape[-1]
            weight = tf.tile(self.filt, (1, 1, 1, self.channels))
        else:
            channels = self.channels
            weight = self.filt
        return nn.conv2d_func(x, weight, strides=self.stride, groups=channels)


def create_aa(
        aa_layer,
        channels: Optional[int] = None,
        stride: int = 2,
        enable: bool = True,
        noop = identity
):
    """ Anti-aliasing """
    if not aa_layer or not enable:
        return noop() if noop is not None else None

    if isinstance(aa_layer, str):
        aa_layer = aa_layer.lower().replace('_', '').replace('-', '')
        if aa_layer == 'avg' or aa_layer == 'avgpool':
            aa_layer = nn.avg_pool2d
        elif aa_layer == 'blur' or aa_layer == 'blurpool':
            aa_layer = BlurPool2d
        elif aa_layer == 'blurpc':
            aa_layer = partial(BlurPool2d, pad_mode='CONSTANT')

        else:
            assert False, f"Unknown anti-aliasing layer ({aa_layer})."

    try:
        return aa_layer(channels=channels, stride=stride)
    except TypeError as e:
        return aa_layer(stride)
