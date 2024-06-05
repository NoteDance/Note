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
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim


def nchw_to(x, fmt: Format):
    if fmt == Format.NHWC:
        x = tf.transpose(x, (0, 2, 3, 1))
    elif fmt == Format.NLC:
        N, C, H, W = x.shape
        x = tf.transpose(tf.reshape(x, (N, C, -1)), (0, 2, 1))
    elif fmt == Format.NCL:
        N, C, H, W = x.shape
        x = tf.reshape(x, (N, C, -1))
    return x


def nhwc_to(x, fmt: Format):
    if fmt == Format.NCHW:
        x = tf.transpose(x, (0, 3, 1, 2))
    elif fmt == Format.NLC:
        N, H, W, C = x.shape
        x = tf.reshape(x, (N, -1, C))
    elif fmt == Format.NCL:
        N, H, W, C = x.shape
        x = tf.transpose(tf.reshape(x, (N, -1, C)), (0, 2, 1))
    return x