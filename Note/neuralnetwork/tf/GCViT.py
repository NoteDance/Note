#!/usr/bin/env python3

# Copyright (c) 2024, NoteDance.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# written by NoteDance


import tensorflow as tf
from Note import nn


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)

    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size, h_w, w_w):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, h_w, window_size, w_w, window_size, C))
    windows = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W, h_w, w_w, B):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] // (H * W // window_size // window_size))
    x = tf.reshape(windows, (B, h_w, w_w, window_size, window_size, -1))
    x = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (B, H, W, -1))
    return x


class Mlp:
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=tf.nn.gelu,
                 drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.dense(hidden_features, in_features)
        self.act = act_layer
        self.fc2 = nn.dense(out_features, hidden_features)
        self.drop = nn.dropout(drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE:
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        self.avg_pool = nn.adaptive_avg_pooling2d(1)
        self.fc = nn.Sequential()
        self.fc.add(nn.dense(int(inp * expansion), oup, use_bias=False))
        self.fc.add(tf.nn.gelu)
        self.fc.add(nn.dense(oup, int(inp * expansion), use_bias=False))
        self.fc.add(tf.nn.sigmoid)

    def __call__(self, x):
        b, _, _, c = x.shape
        y = tf.reshape(self.avg_pool(x), (b, c))
        y = tf.reshape(self.fc(y), (b, 1, 1, c))
        return x * y


class ReduceSize:
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 norm_layer=nn.layer_norm,
                 keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        self.conv = nn.Sequential()
        self.conv.add(nn.conv2d(dim, 3, dim, 1, 1,
                        groups=dim, use_bias=False))
        self.conv.add(tf.nn.gelu)
        self.conv.add(SE(dim, dim))
        self.conv.add(nn.conv2d(dim, 1, dim, 1, 0, use_bias=False))
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.conv2d(dim_out, 3, dim, 2, 1, use_bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def __call__(self, x):
        x = self.norm1(x)
        x = x + self.conv(x)
        x = self.reduction(x)
        x = self.norm2(x)
        return x


class PatchEmbed:
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, in_chans=3, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """

        self.proj = nn.conv2d(dim, 3, in_chans, 2, 1)
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)

    def __call__(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class FeatExtract:
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        self.conv = nn.Sequential()
        self.conv.add(nn.conv2d(dim, 3, dim, 1, 1,
                        groups=dim, use_bias=False))
        self.conv.add(tf.nn.gelu)
        self.conv.add(SE(dim, dim))
        self.conv.add(nn.conv2d(dim, 1, dim, 1, 0, use_bias=False))
        if not keep_dim:
            self.pool = nn.max_pool2d(kernel_size=3, strides=2, padding=1)
        self.keep_dim = keep_dim

    def __call__(self, x):
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class WindowAttention:
    """
    Local window attention based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = tf.math.floordiv(dim, num_heads)
        self.scale = qk_scale or tf.cast(head_dim, 'float32') ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            tf.zeros(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)))
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w))
        coords_flatten = tf.reshape(coords, [coords.shape[0], -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.Variable(tf.transpose(relative_coords, (1, 2, 0)))
        relative_coords[:, :, 0].assign(relative_coords[:, :, 0] + (self.window_size[0] - 1))
        relative_coords[:, :, 1].assign(relative_coords[:, :, 1] + (self.window_size[1] - 1))
        relative_coords[:, :, 0].assign(relative_coords[:, :, 0] * (2 * self.window_size[1] - 1))
        self.relative_position_index = tf.reduce_sum(relative_coords, axis=-1)
        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

        nn.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = tf.nn.softmax

    def __call__(self, x, q_global):
        B_, N, C = x.shape
        head_dim = tf.math.floordiv(C, self.num_heads)
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B_, N, 3, self.num_heads, head_dim)), 
                           (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
        relative_position_bias = \
            tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1))), (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, 0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionGlobal:
    """
    Global window attention based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = tf.math.floordiv(dim, num_heads)
        self.scale = qk_scale or tf.cast(head_dim, 'float32') ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            tf.zeros(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)))
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w))
        coords_flatten = tf.reshape(coords, [coords.shape[0], -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.Variable(tf.transpose(relative_coords, (1, 2, 0)))
        relative_coords[:, :, 0].assign(relative_coords[:, :, 0] + (self.window_size[0] - 1))
        relative_coords[:, :, 1].assign(relative_coords[:, :, 1] + (self.window_size[1] - 1))
        relative_coords[:, :, 0].assign(relative_coords[:, :, 0] * (2 * self.window_size[1] - 1))
        self.relative_position_index = tf.reduce_sum(relative_coords, axis=-1)
        self.qkv = nn.dense(dim * 2, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)
        nn.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = tf.nn.softmax

    def __call__(self, x, q_global):
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = tf.math.floordiv(C, self.num_heads)
        B_dim = tf.math.floordiv(B_, B)
        kv = tf.transpose(tf.reshape(self.qkv(x), (B_, N, 2, self.num_heads, head_dim)), 
                          (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]
        q_global = tf.tile(q_global, (1, B_dim, 1, 1, 1))
        q = tf.reshape(q_global, (B_, self.num_heads, N, head_dim))
        q = q * self.scale
        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
        relative_position_bias = \
            tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1))), (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, 0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GCViTBlock:
    """
    GCViT block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=tf.nn.gelu,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.layer_norm,
                 layer_scale=None,
                 ):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            num_heads: number of attention head.
            window_size: window size.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            act_layer: activation function.
            attention: attention block type.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              )

        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * tf.ones((dim)))
            self.gamma2 = nn.Parameter(layer_scale * tf.ones((dim)))
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        inp_w = tf.math.floordiv(input_resolution, window_size)
        self.num_windows = int(inp_w * inp_w)

    def __call__(self, x, q_global):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        h_w = tf.math.floordiv(H, self.window_size)
        w_w = tf.math.floordiv(W, self.window_size)
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, C))
        attn_windows = self.attn(x_windows, q_global)
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w, B)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GlobalQueryGen:
    """
    Global query generator based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 image_resolution,
                 window_size,
                 num_heads):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.

        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2. Please check Fig.5 of GC ViT paper for details.
        """

        if input_resolution == image_resolution//4:
            self.to_q_global = nn.Sequential()
            self.to_q_global.add(FeatExtract(dim, keep_dim=False))
            self.to_q_global.add(FeatExtract(dim, keep_dim=False))
            self.to_q_global.add(FeatExtract(dim, keep_dim=False))
            
        elif input_resolution == image_resolution//8:
            self.to_q_global = nn.Sequential()
            self.to_q_global.add(FeatExtract(dim, keep_dim=False))
            self.to_q_global.add(FeatExtract(dim, keep_dim=False))

        elif input_resolution == image_resolution//16:

            if window_size == input_resolution:
                self.to_q_global = nn.Sequential()
                self.to_q_global.add(FeatExtract(dim, keep_dim=True))

            else:
                self.to_q_global = nn.Sequential()
                self.to_q_global.add(FeatExtract(dim, keep_dim=True))

        elif input_resolution == image_resolution//32:
            self.to_q_global = nn.Sequential()
            self.to_q_global.add(FeatExtract(dim, keep_dim=True))

        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = tf.math.floordiv(dim, self.num_heads)

    def __call__(self, x):
        x = self.to_q_global(x)
        B = x.shape[0]
        x = tf.transpose(tf.reshape(x, (B, 1, self.N, self.num_heads, self.dim_head)), 
                         (0, 1, 3, 2, 4))
        return x


class GCViTLayer:
    """
    GCViT layer based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 image_resolution,
                 num_heads,
                 window_size,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.layer_norm,
                 layer_scale=None):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            input_resolution: input image resolution.
            window_size: window size in each stage.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        self.blocks = [
            GCViTBlock(dim=dim,
                       num_heads=num_heads,
                       window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                       drop=drop,
                       attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer,
                       layer_scale=layer_scale,
                       input_resolution=input_resolution)
            for i in range(depth)]
        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        self.q_global_gen = GlobalQueryGen(dim, input_resolution, image_resolution, window_size, num_heads)

    def __call__(self, x):
        q_global = self.q_global_gen(x)
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)


class GCViT(nn.Model):
    """
    GCViT based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.layer_norm,
                 layer_scale=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            resolution: input image resolution.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        super().__init__()
        nn.Model.add()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.dropout(drop_rate)
        dpr = [x for x in tf.linspace(0., drop_path_rate, sum(depths))]
        self.levels = []
        for i in range(len(depths)):
            level = GCViTLayer(dim=int(dim * 2 ** i),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < len(depths) - 1),
                               layer_scale=layer_scale,
                               input_resolution=int(2 ** (-2 - i) * resolution),
                               image_resolution=resolution)
            self.levels.append(level)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.adaptive_avg_pooling2d(1)
        self.head = nn.dense(num_classes, num_features) if num_classes > 0 else nn.identity()
        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            nn.trunc_normal_(l.weight, std=.02)

    def no_weight_decay_keywords(self):
        return ['rpb']

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = tf.reshape(x, (x.shape[0], -1))
        return x

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def gc_vit_xxtiny(**kwargs) -> GCViT:
    model = GCViT(depths=[2, 2, 6, 2],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=0.2,
                        **kwargs)
    return model


def gc_vit_xtiny(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 6, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=0.2,
                        **kwargs)
    return model


def gc_vit_tiny(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 19, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=0.2,
                        **kwargs)
    return model


def gc_vit_tiny2(pretrained=False, **kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 29, 5],
                        num_heads=[2, 4, 8, 16],
                        window_size=[7, 7, 14, 7],
                        dim=64,
                        mlp_ratio=3,
                        drop_path_rate=0.25,
                        **kwargs)
    return model


def gc_vit_small(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 19, 5],
                        num_heads=[3, 6, 12, 24],
                        window_size=[7, 7, 14, 7],
                        dim=96,
                        mlp_ratio=2,
                        drop_path_rate=0.3,
                        layer_scale=1e-5,
                        **kwargs)
    return model


def gc_vit_small2(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 23, 5],
                        num_heads=[3, 6, 12, 24],
                        window_size=[7, 7, 14, 7],
                        dim=96,
                        mlp_ratio=3,
                        drop_path_rate=0.35,
                        layer_scale=1e-5,
                        **kwargs)
    return model


def gc_vit_base(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 19, 5],
                        num_heads=[4, 8, 16, 32],
                        window_size=[7, 7, 14, 7],
                        dim=128,
                        mlp_ratio=2,
                        drop_path_rate=0.5,
                        layer_scale=1e-5,
                        **kwargs)
    return model


def gc_vit_large(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[7, 7, 14, 7],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=0.5,
                        layer_scale=1e-5,
                        **kwargs)
    return model


def gc_vit_large_224_21k(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[7, 7, 14, 7],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=0.5,
                        layer_scale=1e-5,
                        **kwargs)
    return model


def gc_vit_large_384_21k(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[12, 12, 24, 12],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=0.1,
                        layer_scale=1e-5,
                        **kwargs)
    return model


def gc_vit_large_512_21k(**kwargs) -> GCViT:
    model = GCViT(depths=[3, 4, 19, 5],
                        num_heads=[6, 12, 24, 48],
                        window_size=[16, 16, 32, 16],
                        dim=192,
                        mlp_ratio=2,
                        drop_path_rate=0.1,
                        layer_scale=1e-5,
                        **kwargs)
    return model