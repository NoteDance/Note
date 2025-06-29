from enum import Enum
from typing import Union

import tensorflow as tf


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

FormatT = Union[str, Format]


def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        return (1,)
    elif fmt is Format.NCL:
        return (2,)
    elif fmt is Format.NHWC:
        return (1, 2)
    else:
        return (2, 3)


def get_channel_dim(fmt: FormatT) -> int:
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        return 3
    elif fmt is Format.NLC:
        return 2
    else:
        return 1


def nchw_to(x: tf.Tensor, fmt: FormatT) -> tf.Tensor:
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        return tf.transpose(x, [0, 2, 3, 1])
    elif fmt is Format.NLC:
        N, C, H, W = x.shape
        flat = tf.reshape(x, [N, C, H * W])
        return tf.transpose(flat, [0, 2, 1])
    elif fmt is Format.NCL:
        N, C, H, W = x.shape
        return tf.reshape(x, [N, C, H * W])
    else:
        return x


def nhwc_to(x: tf.Tensor, fmt: FormatT) -> tf.Tensor:
    fmt = Format(fmt)
    if fmt is Format.NCHW:
        return tf.transpose(x, [0, 3, 1, 2])
    elif fmt is Format.NLC:
        N, H, W, C = x.shape
        return tf.reshape(x, [N, H * W, C])
    elif fmt is Format.NCL:
        N, H, W, C = x.shape
        flat = tf.reshape(x, [N, H * W, C])
        return tf.transpose(flat, [0, 2, 1])
    else:
        return x