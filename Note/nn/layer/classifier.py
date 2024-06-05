""" Classifier head and layer factory

Hacked together by / Copyright 2024 NoteDance
"""
from functools import partial
from typing import Optional, Union, Callable

import tensorflow as tf
from Note import nn
from Note.nn.layer import layer_norm

from .adaptive_avgmax_pool import SelectAdaptivePool2d


def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.conv2d(num_classes, 1, num_features, use_bias=True)
    else:
        fc = nn.dense(num_classes, num_features, use_bias=True)
    return fc


def create_classifier(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: str = 'NHWC',
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    if drop_rate is not None:
        dropout = nn.dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc


class ClassifierHead:
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False,
            input_fmt: str = 'NHWC',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.drop = nn.dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.flatten() if use_conv and pool_type else nn.identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.flatten() if self.use_conv and pool_type else nn.identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def __call__(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


class NormMlpClassifierHead:

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            hidden_size: Optional[int] = None,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            norm_layer: Union[str, Callable] = layer_norm,
            act_layer: Union[str, Callable] = tf.nn.tanh,
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = not pool_type
        norm_layer = norm_layer
        act_layer = act_layer
        linear_layer = partial(nn.conv2d, kernel_size=1) if self.use_conv else nn.dense

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.flatten() if pool_type else nn.identity()
        if hidden_size:
            self.pre_logits = nn.Layers()
            self.pre_logits.add(linear_layer(hidden_size, input_size=in_features))
            self.pre_logits.add(act_layer())
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.identity()
        self.drop = nn.dropout(drop_rate)
        self.fc = linear_layer(num_classes, input_size=self.num_features) if num_classes > 0 else nn.identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
            self.flatten = nn.flatten() if pool_type else nn.identity()
        self.use_conv = self.global_pool.is_identity()
        linear_layer = partial(nn.conv2d, kernel_size=1) if self.use_conv else nn.dense
        if self.hidden_size:
            if ((isinstance(self.pre_logits.fc, nn.conv2d) and not self.use_conv) or
                    (isinstance(self.pre_logits.fc, nn.dense) and self.use_conv)):
                with tf.stop_gradient():
                    new_fc = linear_layer(self.hidden_size, input_size=self.in_features)
                    new_fc.weight.assign(tf.reshape(self.pre_logits.fc.weight, (new_fc.weight.shape)))
                    new_fc.bias.assign(self.pre_logits.fc.bias)
                    self.pre_logits.fc = new_fc
        self.fc = linear_layer(num_classes, input_size=self.num_features) if num_classes > 0 else nn.identity()

    def __call__(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x