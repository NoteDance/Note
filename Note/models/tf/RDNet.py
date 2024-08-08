"""
RDNet
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

from typing import List

import tensorflow as tf
from Note import nn


class RDNetClassifierHead:
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.,
    ):
        self.in_features = in_features
        self.num_features = in_features

        self.norm = nn.layer_norm(in_features)
        self.drop = nn.dropout(drop_rate)
        self.fc = nn.dense(num_classes, self.num_features) if num_classes > 0 else nn.identity()

    def reset(self, num_classes):
        self.fc = nn.dense(num_classes, self.num_features) if num_classes > 0 else nn.identity()

    def __call__(self, x, pre_logits: bool = False):
        x = tf.reduce_mean(x, axis=[-2, -1])
        x = self.norm(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class PatchifyStem:
    def __init__(self, num_input_channels, num_init_features, patch_size=4):

        self.stem = nn.Sequential()
        self.stem.add(nn.conv2d(num_init_features, patch_size, num_input_channels, strides=patch_size))
        self.stem.add(nn.layer_norm(num_init_features))

    def __call__(self, x):
        return self.stem(x)


class Block:
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        self.layers = nn.Sequential()
        self.layers.add(nn.conv2d(in_chs, 7, in_chs, groups=in_chs, strides=1, padding=3))
        self.layers.add(nn.layer_norm(in_chs, epsilon=1e-6))
        self.layers.add(nn.conv2d(inter_chs, 1, in_chs, strides=1, padding=0))
        self.layers.add(tf.nn.gelu)
        self.layers.add(nn.conv2d(out_chs, 1, inter_chs, strides=1, padding=0))

    def __call__(self, x):
        return self.layers(x)


class BlockESE:
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        self.layers = nn.Sequential()
        self.layers.add(nn.conv2d(in_chs, 7, in_chs, groups=in_chs, strides=1, padding=3))
        self.layers.add(nn.layer_norm(in_chs, epsilon=1e-6))
        self.layers.add(nn.Conv2d(inter_chs, 1, in_chs, strides=1, padding=0))
        self.layers.add(tf.nn.gelu)
        self.layers.add(nn.conv2d(out_chs, 1, inter_chs, strides=1, padding=0))
        self.layers.add(nn.EffectiveSEModule(out_chs))

    def __call__(self, x):
        return self.layers(x)


class DenseBlock:
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bottleneck_width_ratio,
        drop_path_rate,
        drop_rate=0.0,
        rand_gather_step_prob=0.0,
        block_idx=0,
        block_type="Block",
        ls_init_value=1e-6,
        **kwargs,
    ):
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.rand_gather_step_prob = rand_gather_step_prob
        self.block_idx = block_idx
        self.growth_rate = growth_rate

        self.gamma = nn.Parameter(ls_init_value * tf.ones(growth_rate)) if ls_init_value > 0 else None
        growth_rate = int(growth_rate)
        inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8

        if self.drop_path_rate > 0:
            self.drop_path = nn.stochastic_depth(drop_path_rate)

        self.layers = eval(block_type)(
            in_chs=num_input_features,
            inter_chs=inter_chs,
            out_chs=growth_rate,
        )
        nn.Model.layer_list.append(self)
        self.training = True

    def __call__(self, x):
        if isinstance(x, List):
            x = tf.concat(x, axis=-1)
        x = self.layers(x)

        if self.gamma is not None:
            x = x * tf.reshape(self.gamma, (1, 1, 1, -1))

        if self.drop_path_rate > 0 and self.training:
            x = self.drop_path(x)
        return x


class DenseStage(nn.Sequential):
    def __init__(self, num_block, num_input_features, drop_path_rates, growth_rate, **kwargs):
        super().__init__()
        for i in range(num_block):
            layer = DenseBlock(
                num_input_features=num_input_features,
                growth_rate=growth_rate,
                drop_path_rate=drop_path_rates[i],
                block_idx=i,
                **kwargs,
            )
            num_input_features += growth_rate
            self.add(layer)
        self.num_out_features = num_input_features

    def __call__(self, init_feature):
        features = [init_feature]
        for module in self.layer:
            new_feature = module(features)
            features.append(new_feature)
        return tf.concat(features, axis=-1)


class RDNet(nn.Model):
    def __init__(
        self,
        num_init_features=64,
        growth_rates=(64, 104, 128, 128, 128, 128, 224),
        num_blocks_list=(3, 3, 3, 3, 3, 3, 3),
        bottleneck_width_ratio=4,
        zero_head=False,
        in_chans=3,  # timm option [--in-chans]
        num_classes=1000,  # timm option [--num-classes]
        drop_rate=0.0,  # timm option [--drop: dropout ratio]
        drop_path_rate=0.0,  # timm option [--drop-path: drop-path ratio]
        checkpoint_path=None,  # timm option [--initial-checkpoint]
        transition_compression_ratio=0.5,
        ls_init_value=1e-6,
        is_downsample_block=(None, True, True, False, False, False, True),
        block_type="Block",
        head_init_scale: float = 1.,
        **kwargs,
    ):
        super().__init__()
        nn.Model.add()
        assert len(growth_rates) == len(num_blocks_list) == len(is_downsample_block)

        self.num_classes = num_classes
        if isinstance(block_type, str):
            block_type = [block_type] * len(growth_rates)

        # stem
        self.stem = PatchifyStem(in_chans, num_init_features, patch_size=4)

        # features
        self.feature_info = []
        self.num_stages = len(growth_rates)
        curr_stride = 4  # stem_stride
        num_features = num_init_features
        dp_rates = [x.numpy().tolist() for x in tf.split(tf.linspace(0., drop_path_rate, sum(num_blocks_list)), num_blocks_list)]

        dense_stages = []
        for i in range(self.num_stages):
            dense_stage_layers = []
            if i != 0:
                compressed_num_features = int(num_features * transition_compression_ratio / 8) * 8
                k_size = stride = 1
                if is_downsample_block[i]:
                    curr_stride *= 2
                    k_size = stride = 2
                dense_stage_layers.append(nn.layer_norm(num_features))
                dense_stage_layers.append(
                    nn.conv2d(compressed_num_features, k_size, num_features, strides=stride, padding=0)
                )
                num_features = compressed_num_features

            stage = DenseStage(
                num_block=num_blocks_list[i],
                num_input_features=num_features,
                growth_rate=growth_rates[i],
                bottleneck_width_ratio=bottleneck_width_ratio,
                drop_rate=drop_rate,
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                block_type=block_type[i],
            )
            dense_stage_layers.append(stage)
            num_features += num_blocks_list[i] * growth_rates[i]

            if i + 1 == self.num_stages or (i + 1 != self.num_stages and is_downsample_block[i + 1]):
                self.feature_info += [
                    dict(
                        num_chs=num_features,
                        reduction=curr_stride,
                        module=f'dense_stages.{i}',
                        growth_rate=growth_rates[i],
                    )
                ]
            layers = nn.Sequential()
            layers.add(dense_stage_layers)
            dense_stages.append(layers)
        layers = nn.Sequential()
        layers.add(dense_stages)
        self.dense_stages = layers

        # classifier
        self.head = RDNetClassifierHead(num_features, num_classes, drop_rate=drop_rate)

        # initialize weights
        nn.Model.apply(self.init_weights)
        self.head.fc.weight.assign(self.head.fc.weight * head_init_scale)
        self.head.fc.bias.assign(self.head.fc.bias * head_init_scale)

        if zero_head:
            self.head.fc.weight.assign(self.head.fc.weight * 0.)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        assert global_pool is None
        self.head.reset(num_classes)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.dense_stages(x)
        return x

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def group_matcher(self, coarse=False):
        assert not coarse
        return dict(
            stem=r'^stem',
            blocks=r'^dense_stages\.(\d+)',
        )

    def init_weights(self, l):
        if isinstance(l, nn.conv2d):
            l.weight.assign(nn.kaiming_normal_(l.weight))


def rdnet_tiny():
    n_layer = 7
    model = RDNet(num_init_features = 64,
            growth_rates = [64] + [104] + [128] * 4 + [224],
            num_blocks_list = [3] * n_layer,
            is_downsample_block = (None, True, True, False, False, False, True),
            transition_compression_ratio = 0.5,
            block_type = ["Block"] + ["Block"] + ["BlockESE"] * 4 + ["BlockESE"]
            )
    return model


def rdnet_small():
    n_layer = 11
    model = RDNet(num_init_features = 72,
            growth_rates = [64] + [128] + [128] * (n_layer - 4) + [240] * 2,
            num_blocks_list = [3] * n_layer,
            is_downsample_block = (None, True, True, False, False, False, False, False, False, True, False),
            transition_compression_ratio = 0.5,
            block_type = ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2
            )
    return model


def rdnet_base():
    n_layer = 11
    model = RDNet(num_init_features = 120,
            growth_rates = [96] + [128] + [168] * (n_layer - 4) + [336] * 2,
            num_blocks_list = [3] * n_layer,
            is_downsample_block = (None, True, True, False, False, False, False, False, False, True, False),
            transition_compression_ratio = 0.5,
            block_type = ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2
            )
    return model


def rdnet_large():
    n_layer = 12
    model = RDNet(num_init_features = 144,
            growth_rates = [128] + [192] + [256] * (n_layer - 4) + [360] * 2,
            num_blocks_list = [3] * n_layer,
            is_downsample_block = (None, True, True, False, False, False, False, False, False, False, True, False),
            transition_compression_ratio = 0.5,
            block_type = ["Block"] + ["Block"] + ["BlockESE"] * (n_layer - 4) + ["BlockESE"] * 2
            )
    return model