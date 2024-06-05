""" PyTorch selectable adaptive pooling
Adaptive pooling with the ability to select the type of pooling from:
    * 'avg' - Average pooling
    * 'max' - Max pooling
    * 'avgmax' - Sum of average and max pooling re-scaled by 0.5
    * 'avgmaxc' - Concatenation of average and max pooling along feature dim, doubles feature dim

Both a functional and a nn.Module version of the pooling is provided.

Hacked together by / Copyright 2024 NoteDance
"""
from typing import Tuple, Union

import tensorflow as tf
from Note import nn

from .format import get_spatial_dim, get_channel_dim

_int_tuple_2_t = Union[int, Tuple[int, int]]


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    x_avg = nn.adaptive_avg_pooling2d(output_size)(x)
    x_max = nn.adaptive_max_pooling2d(output_size)(x)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    x_avg = nn.adaptive_avg_pooling2d(output_size)(x)
    x_max = nn.adaptive_max_pooling2d(output_size)(x)
    return tf.concat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size: _int_tuple_2_t = 1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = nn.adaptive_avg_pooling2d(output_size)(x)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = nn.adaptive_max_pooling2d(output_size)(x)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class FastAdaptiveAvgPool:
    def __init__(self, flatten: bool = False, input_fmt = 'NHWC'):
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def __call__(self, x):
        return tf.reduce_mean(x, self.dim, keepdims=not self.flatten)


class FastAdaptiveMaxPool:
    def __init__(self, flatten: bool = False, input_fmt: str = 'NHWC'):
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def __call__(self, x):
        return tf.reduce_max(x, self.dim, keepdims=not self.flatten)


class FastAdaptiveAvgMaxPool:
    def __init__(self, flatten: bool = False, input_fmt: str = 'NHWC'):
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def __call__(self, x):
        x_avg = tf.reduce_mean(x, self.dim, keepdim=not self.flatten)
        x_max = tf.reduce_max(x, self.dim, keepdim=not self.flatten)
        return 0.5 * x_avg + 0.5 * x_max


class FastAdaptiveCatAvgMaxPool:
    def __init__(self, flatten: bool = False, input_fmt: str = 'NHWC'):
        self.flatten = flatten
        self.dim_reduce = get_spatial_dim(input_fmt)
        if flatten:
            self.dim_cat = 1
        else:
            self.dim_cat = get_channel_dim(input_fmt)

    def __call__(self, x):
        x_avg = tf.reduce_mean(x, self.dim_reduce, keepdim=not self.flatten)
        x_max = tf.reduce_max(x, self.dim_reduce, keepdim=not self.flatten)
        return tf.concat((x_avg, x_max), self.dim_cat)


class AdaptiveAvgMaxPool2d:
    def __init__(self, output_size: _int_tuple_2_t = 1):
        self.output_size = output_size

    def __call__(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d:
    def __init__(self, output_size: _int_tuple_2_t = 1):
        self.output_size = output_size

    def __call__(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d:
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NHWC',
    ):
        assert input_fmt in ('NCHW', 'NHWC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        pool_type = pool_type.lower()
        if not pool_type:
            self.pool = nn.identity()  # pass through
            self.flatten = nn.flatten() if flatten else nn.identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHW':
            assert output_size == 1, 'Fast pooling and non NCHW input formats require output_size == 1.'
            if pool_type.endswith('catavgmax'):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('avgmax'):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type.endswith('max'):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type == 'fast' or pool_type.endswith('avg'):
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            else:
                assert False, 'Invalid pool type: %s' % pool_type
            self.flatten = nn.identity()
        else:
            assert input_fmt == 'NCHW'
            if pool_type == 'avgmax':
                self.pool = AdaptiveAvgMaxPool2d(output_size)
            elif pool_type == 'catavgmax':
                self.pool = AdaptiveCatAvgMaxPool2d(output_size)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            elif pool_type == 'avg':
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            else:
                assert False, 'Invalid pool type: %s' % pool_type
            self.flatten = nn.flatten() if flatten else nn.identity()

    def is_identity(self):
        return not self.pool_type

    def __call__(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'