# Copyright (c) 2024-present, NoteDance, Inc.
# All rights reserved.
"""
Implementation of Cross-Covariance Image Transformer (XCiT)
"""
import math

import tensorflow as tf
from Note import nn
from functools import partial


class PositionalEncodingFourier:
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        self.token_projection = nn.conv2d(dim, 1, hidden_dim * 2)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def __call__(self, B, H, W):
        mask = tf.zeros((B, H, W), dtype=tf.bool)
        not_mask = ~mask
        not_mask = tf.cast(not_mask, dtype=tf.float32)
        y_embed = tf.math.cumsum(not_mask, axis=1)
        x_embed = tf.math.cumsum(not_mask, axis=2)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = tf.range(self.hidden_dim, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = tf.stack((tf.sin(pos_x[:, :, :, 0::2]),
                             tf.cos(pos_x[:, :, :, 1::2])), axis=4)
        pos_x = tf.reshape(pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], -1))
        pos_y = tf.stack((tf.sin(pos_y[:, :, :, 0::2]),
                             tf.cos(pos_y[:, :, :, 1::2])), axis=4)
        pos_y = tf.reshape(pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], -1))
        pos = tf.concat((pos_y, pos_x), axis=3)
        pos = self.token_projection(pos)
        return pos


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    layers = nn.Layers()
    layers.add(nn.zeropadding2d(padding=1))
    layers.add(nn.conv2d(out_planes, 3, in_planes, strides=stride, use_bias=False))
    layers.add(nn.batch_norm(out_planes, synchronized=True))
    return layers


class ConvPatchEmbed:
    """ Image to Patch Embedding using multiple convolutional layers
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        img_size = nn.to_2tuple(img_size)
        patch_size = nn.to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = nn.Layers()
            self.proj.add(conv3x3(3, embed_dim // 8, 2))
            self.proj.add(tf.nn.gelu)
            self.proj.add(conv3x3(embed_dim // 8, embed_dim // 4, 2))
            self.proj.add(tf.nn.gelu)
            self.proj.add(conv3x3(embed_dim // 4, embed_dim // 2, 2))
            self.proj.add(tf.nn.gelu)
            self.proj.add(conv3x3(embed_dim // 2, embed_dim, 2))
        elif patch_size[0] == 8:
            self.proj = nn.Layers()
            self.proj.add(conv3x3(3, embed_dim // 4, 2))
            self.proj.add(tf.nn.gelu)
            self.proj.add(conv3x3(embed_dim // 4, embed_dim // 2, 2))
            self.proj.add(tf.nn.gelu)
            self.proj.add(conv3x3(embed_dim // 2, embed_dim, 2))
        else:
            raise("For convolutional projection, patch size has to be in [8, 16]")

    def __call__(self, x, padding_size=None):
        B, H, W, C = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[1], x.shape[2]
        x = tf.reshape(x, (B, Hp*Wp, -1))

        return x, (Hp, Wp)


class LPI:
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu,
                 drop=0., kernel_size=3):
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = nn.group_conv2d(out_features, kernel_size, out_features,
                                     in_features)
        self.act = act_layer
        self.bn = nn.batch_norm(in_features, synchronized=True)
        self.conv2 = nn.group_conv2d(out_features, kernel_size, out_features,
                                     in_features)
        
        self.zeropadding2d = nn.zeropadding2d(padding=padding)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        x = tf.reshape(tf.transpose(x, (0, 2, 1)), (B, H, W, C))
        x = self.zeropadding2d(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.zeropadding2d(x)
        x = self.conv2(x)
        x = tf.reshape(x, (B, N, C))

        return x


class ClassAttention:
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        qc = q[:, :, 0:1]   # CLS token
        attn_cls = tf.reduce_sum((qc * k), axis=-1) * self.scale
        attn_cls = tf.nn.softmax(attn_cls, axis=-1)
        attn_cls = self.attn_drop(attn_cls)

        cls_tkn = tf.reshape(tf.transpose(tf.matmul(tf.expand_dims(attn_cls, 2), v), (0, 2, 1, 3)), (B, 1, C))
        cls_tkn = self.proj(cls_tkn)
        x = tf.concat([self.proj_drop(cls_tkn), x[:, 1:]], axis=1)
        return x


class ClassAttentionBlock:
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm, eta=None,
                 tokens_norm=False):
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        if eta is not None:     # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * tf.ones(dim))
            self.gamma2 = nn.Parameter(eta * tf.ones(dim))
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # FIXME: A hack for models pre-trained with layernorm over all the tokens not just the CLS
        self.tokens_norm = tokens_norm

    def __call__(self, x, H, W, mask=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = tf.concat([self.norm2(x[:, 0:1]), x[:, 1:]], axis=1)

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = tf.concat([cls_token, x[:, 1:]], axis=1)
        x = x_res + self.drop_path(x)
        return x


class XCA:
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        self.num_heads = num_heads
        self.temperature = nn.initializer_((num_heads, 1, 1), 'ones', name='temperature')

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = tf.transpose(q, (0, 1, 3, 2))
        k = tf.transpose(k, (0, 1, 3, 2))
        v = tf.transpose(v, (0, 1, 3, 2))

        q = tf.math.l2_normalize(q, axis=-1)
        k = tf.math.l2_normalize(k, axis=-1)

        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.temperature
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 3, 1, 2)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def no_weight_decay(self):
        return ['temperature']


class XCABlock:
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm,
                 num_tokens=196, eta=None):
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.gamma1 = nn.Parameter(eta * tf.ones(dim))
        self.gamma2 = nn.Parameter(eta * tf.ones(dim))
        self.gamma3 = nn.Parameter(eta * tf.ones(dim))

    def __call__(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class XCiT(nn.Model):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 cls_attn_layers=2, use_pos=True, patch_proj='linear', eta=None, tokens_norm=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA
        """
        super().__init__()
        nn.Model.add()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.layer_norm, epsilon=1e-6)

        self.patch_embed = ConvPatchEmbed(img_size=img_size, embed_dim=embed_dim,
                                          patch_size=patch_size)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.initializer_((1, 1, embed_dim), ['truncated_normal', .02], name='cls_token')
        self.pos_drop = nn.dropout(drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = [
            XCABlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, num_tokens=num_patches, eta=eta)
            for i in range(depth)]

        self.cls_attn_blocks = [
            ClassAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                eta=eta, tokens_norm=tokens_norm)
            for i in range(cls_attn_layers)]
        self.norm = norm_layer(embed_dim)
        self.head = self.dense(num_classes, self.num_features) if num_classes > 0 else nn.identity()

        self.pos_embeder = PositionalEncodingFourier(dim=embed_dim)
        self.use_pos = use_pos

        # Classifier head
        nn.Model.apply(self.init_weights)
        
        self.training = True

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))

    def no_weight_decay(self):
        return ['pos_embed', 'cls_token', 'dist_token']

    def forward_features(self, x):
        B, H, W, C = x.shape

        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos:
            pos_encoding = tf.transpose(tf.reshape(self.pos_embeder(B, Hp, Wp), (B, -1, x.shape[1])), (0, 2, 1))
            x = x + pos_encoding

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = tf.tile(self.cls_token, (B, 1, 1))
        x = tf.concat((cls_tokens, x), axis=1)

        for blk in self.cls_attn_blocks:
            x = blk(x, Hp, Wp)

        x = self.norm(x)[:, 0]
        return x

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        if self.training:
            return x, x
        else:
            return x


# Patch size 16x16 models
def xcit_nano_12_p16(**kwargs):
    model = XCiT(
        patch_size=16, embed_dim=128, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1.0, tokens_norm=False, **kwargs)
    return model


def xcit_tiny_12_p16(**kwargs):
    model = XCiT(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    return model


def xcit_small_12_p16(**kwargs):
    model = XCiT(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    return model


def xcit_tiny_24_p16(**kwargs):
    model = XCiT(
        patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model


def xcit_small_24_p16(**kwargs):
    model = XCiT(
        patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model


def xcit_medium_24_p16(**kwargs):
    model = XCiT(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model


def xcit_large_24_p16(**kwargs):
    model = XCiT(
        patch_size=16, embed_dim=768, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model


# Patch size 8x8 models
def xcit_nano_12_p8(**kwargs):
    model = XCiT(
        patch_size=8, embed_dim=128, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1.0, tokens_norm=False, **kwargs)
    return model


def xcit_tiny_12_p8(**kwargs):
    model = XCiT(
        patch_size=8, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    return model


def xcit_small_12_p8(**kwargs):
    model = XCiT(
        patch_size=8, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    return model


def xcit_tiny_24_p8(**kwargs):
    model = XCiT(
        patch_size=8, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model


def xcit_small_24_p8(**kwargs):
    model = XCiT(
        patch_size=8, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model


def xcit_medium_24_p8(**kwargs):
    model = XCiT(
        patch_size=8, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model


def xcit_large_24_p8(**kwargs):
    model = XCiT(
        patch_size=8, embed_dim=768, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    return model