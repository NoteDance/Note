""" Median Pool
Hacked together by / Copyright 2025 NoteDance
"""
import tensorflow as tf
from Note import nn


class MedianPool2d(nn.Layer):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         strides: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, strides=1, padding=0, same=False):
        super().__init__()
        self.k = nn.to_2tuple(kernel_size)
        self.strides = nn.to_2tuple(strides)
        self.padding = nn.to_4tuple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.strides[0] == 0:
                ph = max(self.k[0] - self.strides[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.strides[0]), 0)
            if iw % self.strides[1] == 0:
                pw = max(self.k[1] - self.strides[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.strides[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def __call__(self, x):
        pl, pr, pt, pb = self._padding(x)
        paddings = [[0, 0], [pt, pb], [pl, pr], [0, 0]]
        x = tf.pad(x, paddings, mode='REFLECT')
        x = nn.unfold(nn.unfold(x, 2, self.k[0], self.strides[0]), 3, self.k[1], self.strides[1])
        x = nn.median(tf.reshape(x, tf.concat([tf.shape(x)[:4], [-1]], axis=0)))
        return x