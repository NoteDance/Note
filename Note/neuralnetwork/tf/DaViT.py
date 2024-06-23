""" DaViT: Dual Attention Vision Transformers

As described in https://arxiv.org/abs/2204.03645

Input size invariant transformer architecture that combines channel and spacial
attention in each block. The attention mechanisms used are linear in complexity.

DaViT model defs and weights adapted from https://github.com/dingmyu/davit, original copyright below

"""
# Copyright (c) 2024 NoteDance
# All rights reserved.
# This source code is licensed under the MIT license
from functools import partial
from typing import Optional, Tuple

import tensorflow as tf
from Note import nn


__all__ = ['DaVit']


class ConvPosEnc:
    def __init__(self, dim: int, k: int = 3, act: bool = False):
        self.proj = nn.conv2d(dim, k, dim, 1, k // 2, groups=dim)
        self.act = tf.nn.gelu if act else nn.identity()

    def __call__(self, x):
        feat = self.proj(x)
        x = x + self.act(feat)
        return x


class Stem:
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            stride=4,
            norm_layer=nn.layer_norm,
    ):
        stride = nn.to_2tuple(stride)
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        assert stride[0] == 4  # only setup for stride==4
        self.conv = nn.conv2d(
            out_chs,
            7,
            in_chs,
            strides=stride,
            padding=3,
        )
        self.norm = norm_layer(out_chs)

    def __call__(self, x):
        B, H, W, C = x.shape
        padding_w = (self.stride[1] - W % self.stride[1]) % self.stride[1]
        padding_h = (self.stride[0] - H % self.stride[0]) % self.stride[0]
        x = tf.pad(x, [[0, 0], [0, 0], [0, padding_w], [0, 0]])
        x = tf.pad(x, [[0, 0], [0, padding_h], [0, 0], [0, 0]])
        x = self.conv(x)
        x = self.norm(x)
        return x


class Downsample:
    def __init__(
            self,
            in_chs,
            out_chs,
            norm_layer=nn.layer_norm,
    ):
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.norm = norm_layer(in_chs)
        self.conv = nn.conv2d(
            out_chs,
            2,
            in_chs,
            strides=2,
            padding=0,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        x = self.norm(x)
        padding_w = (2 - W % 2) % 2
        padding_h = (2 - H % 2) % 2
        x = tf.pad(x, [[0, 0], [0, 0], [0, padding_w], [0, 0]])
        x = tf.pad(x, [[0, 0], [0, padding_h], [0, 0], [0, 0]])
        x = self.conv(x)
        return x


class ChannelAttention:

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.proj = nn.dense(dim, dim)

    def __call__(self, x):
        B, N, C = x.shape

        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads)), 
                           (2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv, axis=0)

        k = k * self.scale
        attn = tf.matmul(tf.transpose(k, (0, 1, 3, 2)), v)
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.transpose(tf.matmul(attn, tf.transpose(q, (0, 1, 3, 2))), (0, 1, 3, 2))
        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        return x


class ChannelBlock:

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop_path=0.,
            act_layer=tf.nn.gelu,
            norm_layer=nn.layer_norm,
            ffn=True,
            cpe_act=False,
    ):
        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path1 = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)

        if self.ffn:
            self.norm2 = norm_layer(dim)
            self.mlp = nn.Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
            )
            self.drop_path2 = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        else:
            self.norm2 = None
            self.mlp = None
            self.drop_path2 = None

    def __call__(self, x):
        B, H, W, C = x.shape

        x = self.cpe1(x)
        x = tf.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], -1))

        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path1(cur)

        x = self.cpe2(tf.reshape(x, (B, H, W, C)))

        if self.mlp is not None:
            x = tf.reshape(x, (B, H*W, -1))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            x = tf.reshape(x, (B, H, W, C))

        return x


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C))
    windows = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), 
                         (-1, window_size[0], window_size[1], C))
    return windows


def window_reverse(windows, window_size: Tuple[int, int], H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = tf.reshape(windows, (-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C))
    x = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, H, W, C))
    return x


class WindowAttention:
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, use_fused_attn=True):
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.proj = nn.dense(dim, dim)

        self.softmax = tf.nn.softmax

    def __call__(self, x):
        B_, N, C = x.shape

        qkv = tf.transpose(tf.reshape(self.qkv(x), (B_, N, 3, self.num_heads, C // self.num_heads)), 
                           (2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv, axis=0)

        if self.fused_attn:
            x = nn.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
            attn = self.softmax(attn)
            x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)
        return x


class SpatialBlock:
    r""" Windows Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path=0.,
            act_layer=tf.nn.gelu,
            norm_layer=nn.layer_norm,
            ffn=True,
            cpe_act=False,
    ):
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = nn.to_2tuple(window_size)
        self.mlp_ratio = mlp_ratio

        self.cpe1 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.drop_path1 = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()

        self.cpe2 = ConvPosEnc(dim=dim, k=3, act=cpe_act)
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
            )
            self.drop_path2 = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        else:
            self.norm2 = None
            self.mlp = None
            self.drop_path1 = None

    def __call__(self, x):
        B, H, W, C = x.shape

        shortcut = self.cpe1(x)
        shortcut = tf.reshape(shortcut, (shortcut.shape[0], shortcut.shape[1]*shortcut.shape[2], -1))

        x = self.norm1(shortcut)
        x = tf.reshape(x, (B, H, W, C))

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = tf.pad(x, [[0, 0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]])
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size[0] * self.window_size[1], C))

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size[0], self.window_size[1], C))
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :]

        x = tf.reshape(x, (B, H * W, C))
        x = shortcut + self.drop_path1(x)

        x = self.cpe2(tf.reshape(x, (B, H, W, C)))

        if self.mlp is not None:
            x = tf.reshape(x, (B, H*W, -1))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            x = tf.reshape(x, (B, H, W, C))

        return x


class DaVitStage:
    def __init__(
            self,
            in_chs,
            out_chs,
            depth=1,
            downsample=True,
            attn_types=('spatial', 'channel'),
            num_heads=3,
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rates=(0, 0),
            norm_layer=nn.layer_norm,
            norm_layer_cl=nn.layer_norm,
            ffn=True,
            cpe_act=False
    ):
        # downsample embedding layer at the beginning of each stage
        if downsample:
            self.downsample = Downsample(in_chs, out_chs, norm_layer=norm_layer)
        else:
            self.downsample = nn.identity()

        '''
         repeating alternating attention blocks in each stage
         default: (spatial -> channel) x depth
         
         potential opportunity to integrate with a more general version of ByobNet/ByoaNet
         since the logic is similar
        '''
        stage_blocks = []
        for block_idx in range(depth):
            dual_attention_block = []
            for attn_idx, attn_type in enumerate(attn_types):
                if attn_type == 'spatial':
                    dual_attention_block.append(SpatialBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[block_idx],
                        norm_layer=norm_layer_cl,
                        ffn=ffn,
                        cpe_act=cpe_act,
                        window_size=window_size,
                    ))
                elif attn_type == 'channel':
                    dual_attention_block.append(ChannelBlock(
                        dim=out_chs,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[block_idx],
                        norm_layer=norm_layer_cl,
                        ffn=ffn,
                        cpe_act=cpe_act
                    ))
            layers = nn.Sequential()
            layers.add(dual_attention_block)
            stage_blocks.append(layers)
        self.blocks = nn.Sequential()
        self.blocks.add(stage_blocks)

    def __call__(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class DaVit(nn.Model):
    r""" DaViT
        A PyTorch implementation of `DaViT: Dual Attention Vision Transformers`  - https://arxiv.org/abs/2204.03645
        Supports arbitrary input sizes and pyramid feature extraction
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks in each stage. Default: (1, 1, 3, 1)
        embed_dims (tuple(int)): Patch embedding dimension. Default: (96, 192, 384, 768)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (3, 6, 12, 24)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(
            self,
            in_chans=3,
            depths=(1, 1, 3, 1),
            embed_dims=(96, 192, 384, 768),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.layer_norm,
            norm_layer_cl=nn.layer_norm,
            norm_eps=1e-5,
            attn_types=('spatial', 'channel'),
            ffn=True,
            cpe_act=False,
            drop_rate=0.,
            drop_path_rate=0.,
            num_classes=1000,
            global_pool='avg',
            head_norm_first=False,
    ):
        super().__init__()
        nn.Model.add()
        num_stages = len(embed_dims)
        assert num_stages == len(num_heads) == len(depths)
        norm_layer = partial(norm_layer, epsilon=norm_eps)
        norm_layer_cl = partial(norm_layer_cl, epsilon=norm_eps)
        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = embed_dims[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        self.stem = Stem(in_chans, embed_dims[0], norm_layer=norm_layer)
        in_chs = embed_dims[0]

        linspace_tensor = tf.linspace(0.0, drop_path_rate, tf.reduce_sum(depths))
        split_tensors = tf.split(linspace_tensor, num_or_size_splits=depths)
        dpr = [x.numpy().tolist() for x in split_tensors]
        stages = []
        for stage_idx in range(num_stages):
            out_chs = embed_dims[stage_idx]
            stage = DaVitStage(
                in_chs,
                out_chs,
                depth=depths[stage_idx],
                downsample=stage_idx > 0,
                attn_types=attn_types,
                num_heads=num_heads[stage_idx],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rates=dpr[stage_idx],
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
                ffn=ffn,
                cpe_act=cpe_act,
            )
            in_chs = out_chs
            stages.append(stage)
            self.feature_info += [dict(num_chs=out_chs, reduction=2, module=f'stages.{stage_idx}')]

        self.stages = nn.Sequential()
        self.stages.add(stages)

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default DaViT order, similar to ConvNeXt
        # FIXME generalize this structure to ClassifierHead
        if head_norm_first:
            self.norm_pre = norm_layer(self.num_features)
            self.head = nn.ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.identity()
            self.head = nn.NormMlpClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
            )
        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            nn.trunc_normal_(l.weight, std=.02)

    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,)),
            ]
        )

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def davit_tiny(**kwargs):
    return DaVit(depths=(1, 1, 3, 1), embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24))


def davit_small(**kwargs):
    return DaVit(depths=(1, 1, 9, 1), embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24))


def davit_base(**kwargs):
    return DaVit(depths=(1, 1, 9, 1), embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32))


def davit_large(**kwargs):
    return DaVit(depths=(1, 1, 9, 1), embed_dims=(192, 384, 768, 1536), num_heads=(6, 12, 24, 48))


def davit_huge(**kwargs):
    return DaVit(depths=(1, 1, 9, 1), embed_dims=(256, 512, 1024, 2048), num_heads=(8, 16, 32, 64))


def davit_giant(**kwargs):
    return DaVit(depths=(1, 1, 12, 3), embed_dims=(384, 768, 1536, 3072), num_heads=(12, 24, 48, 96))