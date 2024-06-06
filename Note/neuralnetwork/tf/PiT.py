# PiT
# Copyright 2024-present NoteDance Corp.
# Apache License v2.0

import tensorflow as tf
from einops import rearrange
from Note import nn
import math

from functools import partial

class Transformer:
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = [
            nn.Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.layer_norm, epsilon=1e-6)
            )
            for i in range(depth)]

    def __call__(self, x, cls_tokens):
        h, w = x.shape[1:3]
        x = rearrange(x, 'b h w c -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = tf.concat((cls_tokens, x), axis=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b h w c', h=h, w=w)

        return x, cls_tokens


class conv_head_pooling:
    def __init__(self, in_feature, out_feature, stride,
                 ):
        self.zeropadding2d = nn.zeropadding2d(padding=stride // 2)
        self.conv = nn.group_conv2d(out_feature, stride + 1, in_feature, in_feature,
                              strides=stride,
                              )
        self.fc = nn.dense(out_feature, in_feature)

    def __call__(self, x, cls_token):

        x = self.zeropadding2d(x)
        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding:
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        self.zeropadding2d = nn.zeropadding2d(padding=padding)
        self.conv = nn.conv2d(out_channels, patch_size, in_channels,
                              strides=stride, use_bias=True)

    def __call__(self, x):
        x = self.zeropadding2d(x)
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Model):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super().__init__()
        
        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.initializer_(
            (1, width, width, base_dims[0] * heads[0]), ['truncated_normal', .02], name='pos_embed'
        )
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.cls_token = nn.initializer_(
            (1, 1, base_dims[0] * heads[0]), ['truncated_normal', .02], name='cls_token'
        )
        self.pos_drop = nn.dropout(drop_rate)

        self.transformers = []
        self.pools = []

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.layer_norm(base_dims[-1] * heads[-1], epsilon=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = self.dense(num_classes, base_dims[-1] * heads[-1])
        else:
            self.head = nn.identity()

    def no_weight_decay(self):
        return ['pos_embed', 'cls_token']

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.dense(num_classes, self.embed_dim)
        else:
            self.head = nn.identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = tf.tile(self.cls_token, (x.shape[0], 1, 1))

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def __call__(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.initializer_(
            (1, 2, self.base_dims[0] * self.heads[0]), ['truncated_normal', .02], name='cls_token'
            )
        if self.num_classes > 0:
            self.head_dist = nn.dense(self.num_classes, self.base_dims[-1] * self.heads[-1])
        else:
            self.head_dist = nn.identity()
        
        self.training = True

    def __call__(self, x):
        cls_token = self.forward_features(x)
        x_cls = self.head(cls_token[:, 0])
        x_dist = self.head_dist(cls_token[:, 1])
        if self.training:
            return x_cls, x_dist
        else:
            return (x_cls + x_dist) / 2

def pit_b(**kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    return model

def pit_s(**kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    return model


def pit_xs(**kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return model

def pit_ti(**kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return model


def pit_b_distilled(**kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    return model


def pit_s_distilled(**kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    return model


def pit_xs_distilled(**kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return model


def pit_ti_distilled(**kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return model