""" Split BatchNorm

A PyTorch BatchNorm layer that splits input batch into N equal parts and passes each through
a separate BN layer. The first split is passed through the parent BN layers with weight/bias
keys the same as the original BN. All other splits pass through BN sub-layers under the '.aux_bn'
namespace.

This allows easily removing the auxiliary BN layers after training to efficiently
achieve the 'Auxiliary BatchNorm' as described in the AdvProp Paper, section 4.2,
'Disentangled Learning via An Auxiliary BN'

Hacked together by / Copyright 2025 NoteDance
"""
import tensorflow as tf
from Note import nn


class SplitBatchNorm(nn.batch_norm):

    def __init__(self, num_features, eps=1e-5, momentum=0.9, center=True, scale=True,
                 num_splits=2):
        super().__init__(num_features, epsilon=eps, momentum=momentum, center=center, scale=scale)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = []
        for _ in range(num_splits - 1):
            self.aux_bn.append(nn.batch_norm(num_features, epsilon=eps, momentum=momentum, center=center, scale=scale))
        nn.Model.register(self)

    def __call__(self, input, training=None):
        if training!=None:
            self.training=training
        if self.training:  # aux BN only relevant while training
            split_size = input.shape[0] // self.num_splits
            assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"
            split_input = tf.split(input, split_size)
            x = [super().__call__(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return tf.concat(x, axis=0)
        else:
            return super().__call__(input)


def convert_splitbn_model(module, num_splits=2):
    for layer in module.layer_list:
        if isinstance(layer, nn.batch_norm):
            module.layer_list.remove(layer)
            dict_ = dict(layer.__dict__)
            if dict_['scale']:
                module.param.remove(dict_['gamma'])
            if dict_['center']:
                module.param.remove(dict_['beta'])
            layer.__class__ = SplitBatchNorm
            layer.__init__(dict_['input_size'], dict_['epsilon'], dict_['momentum'], center=dict_['center'], scale=dict_['scale'], num_splits=num_splits)
            layer.moving_mean.assign(dict_['moving_mean'])
            layer.moving_variance.assign(dict_['moving_variance'])
            if dict_['scale']:
                layer.gamma.assign(dict_['gamma'])
            if dict_['center']:
                layer.beta.assign(dict_['beta'])
            for aux in layer.aux_bn:
                aux.moving_mean.assign(dict_['moving_mean'])
                aux.moving_variance.assign(dict_['moving_variance'])
                if dict_['scale']:
                    aux.gamma.assign(dict_['gamma'])
                if dict_['center']:
                    aux.beta.assign(dict_['beta'])
