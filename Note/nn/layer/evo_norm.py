""" EvoNorm in TensorFlow

Based on `Evolving Normalization-Activation Layers` - https://arxiv.org/abs/2004.02967
@inproceedings{NEURIPS2020,
 author = {Liu, Hanxiao and Brock, Andy and Simonyan, Karen and Le, Quoc},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {13539--13550},
 publisher = {Curran Associates, Inc.},
 title = {Evolving Normalization-Activation Layers},
 url = {https://proceedings.neurips.cc/paper/2020/file/9d4c03631b8b0c85ae08bf05eda37d0f-Paper.pdf},
 volume = {33},
 year = {2020}
}

Hacked together by / Copyright 2025 NoteDance
"""
from typing import Sequence, Union

import tensorflow as tf
from Note import nn


def instance_std(x, eps: float = 1e-5):
    std = tf.cast(
        tf.sqrt(
            tf.math.reduce_variance(
                tf.cast(x, tf.float32),
                axis=[1, 2],
                keepdims=True
            ) + eps
        ),
        x.dtype
    )
    return tf.broadcast_to(std, x.shape)


def instance_std_tpu(x, eps: float = 1e-5):
    std = tf.sqrt(manual_var(x, dim=(1, 2)) + eps)
    return tf.broadcast_to(std, x.shape)
# instance_std = instance_std_tpu


def instance_rms(x, eps: float = 1e-5):
    rms = tf.cast(
        tf.math.sqrt(
            tf.reduce_mean(tf.cast(x, tf.float32) ** 2, axis=[1, 2], keepdims=True) + eps
        ),
        x.dtype
    )
    return tf.broadcast_to(rms, tf.shape(x))


def manual_var(x,
               dim: Union[int, Sequence[int]],
               diff_sqm: bool = False):
    xm = tf.reduce_mean(x, axis=dim, keepdims=True)
    if diff_sqm:
        # difference of squared mean and mean squared, faster on TPU can be less stable
        var = tf.clip_by_value(
            tf.reduce_mean(x * x, axis=dim, keepdims=True) - xm * xm,
            clip_value_min=0.0,
            clip_value_max=tf.float64.max
        )
    else:
        var = tf.reduce_mean((x - xm) ** 2, axis=dim, keepdims=True)
    return var


def group_std(x, groups: int = 32, eps: float = 1e-5, flatten: bool = False):
    B, H, W, C = x.shape
    x_dtype = x.dtype
    assert C % groups == 0, ''
    if flatten:
        x = tf.reshape(x, (B, -1, groups))  # FIXME simpler shape causing TPU / XLA issues
        std = tf.cast(
            tf.sqrt(
                tf.math.reduce_variance(
                    tf.cast(x, tf.float32),
                    axis=1,
                    keepdims=True
                ) + eps
            ),
            x_dtype
        )
    else:
        x = tf.reshape(x, (B, H, W, groups, C // groups))
        std = tf.cast(
            tf.sqrt(
                tf.math.reduce_variance(
                    tf.cast(x, tf.float32),
                    axis=[1, 2, 4],
                    keepdims=True
                ) + eps
            ),
            x_dtype
        )
    return tf.reshape(tf.broadcast_to(std, x.shape), (B, H, W, C))


def group_std_tpu(x, groups: int = 32, eps: float = 1e-5, diff_sqm: bool = False, flatten: bool = False):
    # This is a workaround for some stability / odd behaviour of .var and .std
    # running on PyTorch XLA w/ TPUs. These manual var impl are producing much better results
    B, H, W, C = x.shape
    assert C % groups == 0, ''
    if flatten:
        x = tf.reshape(x, (B, -1, groups))  # FIXME simpler shape causing TPU / XLA issues
        var = manual_var(x, dim=-1, diff_sqm=diff_sqm)
    else:
        x = tf.reshape(x, (B, H, W, groups, C // groups))
        var = manual_var(x, dim=(2, 3, 4), diff_sqm=diff_sqm)
    return tf.reshape(tf.broadcast_to(tf.sqrt(var + eps), x.shape), (B, H, W, C))
#group_std = group_std_tpu  # FIXME TPU temporary


def group_rms(x, groups: int = 32, eps: float = 1e-5):
    B, H, W, C = x.shape
    assert C % groups == 0, ''
    x_dtype = x.dtype
    x = tf.reshape(x, (B, H, W, groups, C // groups))
    rms = tf.cast(tf.sqrt(tf.reduce_mean(tf.square(tf.cast(x, tf.float32)), axis=(1, 2, 4), keepdims=True) + eps), x_dtype)
    return tf.reshape(tf.broadcast_to(rms, x.shape), (B, H, W, C))


class EvoNorm2dB0(nn.Layer):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-3, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(tf.ones(num_features))
        self.bias = nn.Parameter(tf.zeros(num_features))
        self.v = nn.Parameter(tf.ones(num_features)) if apply_act else None
        self.running_var = nn.initializer_([num_features], 'ones', trainable=False)
        nn.Model.param.append(self.running_var)
        self.training = True
        nn.Model.layer_list_.append(self)
        if nn.Model.name!=None and nn.Model.name not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name]=[]
            nn.Model.layer_eval[nn.Model.name].append(self)
        elif nn.Model.name!=None:
            nn.Model.layer_eval[nn.Model.name].append(self)

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        if self.v is not None:
            if self.training:
                var = tf.math.reduce_variance(
                        tf.cast(x, tf.float32),
                        axis=[0, 1, 2],
                        keepdims=False
                    )
                # var = tf.squeeze(manual_var(x, dim=(0, 1, 2)))
                n = tf.size(x) / x.shape[-1]
                self.running_var.assign(
                    self.running_var * (1 - self.momentum) +
                    tf.stop_gradient(var) * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            left = tf.broadcast_to(
                    tf.reshape(
                        tf.cast(tf.sqrt(var + self.eps), x_dtype),
                        v_shape
                    ),
                    x.shape
                )
            v = tf.reshape(tf.cast(self.v, x_dtype), v_shape)
            right = x * v + instance_std(x, self.eps)
            x = x / tf.maximum(left, right)
        return x * tf.reshape(tf.cast(self.weight, x_dtype), v_shape) + tf.reshape(tf.cast(self.bias, x_dtype), v_shape)


class EvoNorm2dB1(nn.Layer):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(tf.ones(num_features))
        self.bias = nn.Parameter(tf.zeros(num_features))
        self.running_var = nn.initializer_([num_features], 'ones', trainable=False)
        nn.Model.param.append(self.running_var)
        self.training = True
        nn.Model.layer_list_.append(self)
        if nn.Model.name!=None and nn.Model.name not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name]=[]
            nn.Model.layer_eval[nn.Model.name].append(self)
        elif nn.Model.name!=None:
            nn.Model.layer_eval[nn.Model.name].append(self)

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        if self.apply_act:
            if self.training:
                var = tf.math.reduce_variance(
                        tf.cast(x, tf.float32),
                        axis=[0, 1, 2],
                        keepdims=False
                    )
                n = tf.size(x) / x.shape[-1]
                self.running_var.assign(
                    self.running_var * (1 - self.momentum) +
                    tf.cast(tf.stop_gradient(var), self.running_var.dtype) * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = tf.reshape(tf.cast(var, x_dtype), v_shape)
            left = tf.sqrt(var + self.eps)
            right = (x + 1) * instance_rms(x, self.eps)
            x = x / tf.maximum(left, right)
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)


class EvoNorm2dB2(nn.Layer):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(tf.ones(num_features))
        self.bias = nn.Parameter(tf.zeros(num_features))
        self.running_var = nn.initializer_([num_features], 'ones', trainable=False)
        nn.Model.param.append(self.running_var)
        self.training = True
        nn.Model.layer_list_.append(self)
        if nn.Model.name!=None and nn.Model.name not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name]=[]
            nn.Model.layer_eval[nn.Model.name].append(self)
        elif nn.Model.name!=None:
            nn.Model.layer_eval[nn.Model.name].append(self)

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        if self.apply_act:
            if self.training:
                var = tf.math.reduce_variance(
                        tf.cast(x, tf.float32),
                        axis=[0, 1, 2],
                        keepdims=False
                    )
                n = tf.size(x) / x.shape[-1]
                self.running_var.assign(
                    self.running_var * (1 - self.momentum) +
                    tf.cast(tf.stop_gradient(var), self.running_var.dtype) * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = tf.reshape(tf.cast(var, x_dtype), v_shape)
            left = tf.sqrt(var + self.eps)
            right = instance_rms(x, self.eps) - x
            x = x / tf.maximum(left, right)
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)


class EvoNorm2dS0(nn.Layer):
    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-5, **_):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(tf.ones(num_features))
        self.bias = nn.Parameter(tf.zeros(num_features))
        self.v = nn.Parameter(tf.ones(num_features)) if apply_act else None
        self.training = True
        nn.Model.layer_list_.append(self)
        if nn.Model.name!=None and nn.Model.name not in nn.Model.layer_eval:
            nn.Model.layer_eval[nn.Model.name]=[]
            nn.Model.layer_eval[nn.Model.name].append(self)
        elif nn.Model.name!=None:
            nn.Model.layer_eval[nn.Model.name].append(self)

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        if self.v is not None:
            v = tf.cast(tf.reshape(self.v, v_shape), x_dtype)
            x = x * tf.sigmoid(x * v) / group_std(x, self.groups, self.eps)
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)


class EvoNorm2dS0a(EvoNorm2dS0):
    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, eps=eps)

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        d = group_std(x, self.groups, self.eps)
        if self.v is not None:
            v = tf.cast(tf.resahpe(self.v, v_shape), x_dtype)
            x = x * tf.sigmoid(x * v)
        x = x / d
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)


class EvoNorm2dS1(nn.Layer):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-5, **_):
        super().__init__()
        act_layer = act_layer or tf.nn.silu
        self.apply_act = apply_act  # apply activation (non-linearity)
        if act_layer is not None and apply_act:
            self.act = act_layer
        else:
            self.act = nn.identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.pre_act_norm = False
        self.weight = nn.Parameter(tf.ones(num_features))
        self.bias = nn.Parameter(tf.zeros(num_features))

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        if self.apply_act:
            x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)


class EvoNorm2dS1a(EvoNorm2dS1):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)


class EvoNorm2dS2(nn.Layer):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-5, **_):
        super().__init__()
        act_layer = act_layer or tf.nn.silu
        self.apply_act = apply_act  # apply activation (non-linearity)
        if act_layer is not None and apply_act:
            self.act = act_layer
        else:
            self.act = nn.identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(tf.ones(num_features))
        self.bias = nn.Parameter(tf.zeros(num_features))

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        if self.apply_act:
            x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)


class EvoNorm2dS2a(EvoNorm2dS2):
    def __init__(
            self, num_features, groups=32, group_size=None,
            apply_act=True, act_layer=None, eps=1e-3, **_):
        super().__init__(
            num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def __call__(self, x):
        assert len(x.shape) == 4, 'expected 4D input'
        x_dtype = x.dtype
        v_shape = (1, 1, 1, -1)
        x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * tf.cast(tf.reshape(self.weight, v_shape), x_dtype) + tf.cast(tf.reshape(self.bias, v_shape), x_dtype)