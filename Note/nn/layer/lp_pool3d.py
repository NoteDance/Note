import tensorflow as tf
from Note import nn
from typing import Union
import collections
from itertools import repeat


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_triple = _ntuple(3, "_triple")


class lp_pool3d:
    r"""
    Apply a 3D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.
    
    """
    def __init__(self, norm_type: Union[int, float], kernel_size, strides = None):
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.strides = strides
        if strides is not None:
            self.avg_pool3d = nn.avg_pool3d(kernel_size, strides, 0)
        else:
            self.avg_pool3d = nn.avg_pool3d(kernel_size, padding=0)
    
    def __call__(self, input):
        kd, kw, kh = _triple(self.kernel_size)
        if self.strides is not None:
            out = self.avg_pool3d(tf.pow(input, self.norm_type))
        else:
            out = self.avg_pool3d(tf.pow(input, self.norm_type))
    
        return tf.pow((tf.sign(out) * tf.nn.relu(tf.abs(out))) * (kd * kw * kh), (1.0 / self.norm_type))