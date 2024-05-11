# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.dropout import dropout
from Note.nn.layer.stochastic_depth import stochastic_depth
from Note.nn.layer.identity import identity
from Note.nn.initializer import initializer,initializer_
from Note.nn.Module import Module
from Note.nn.variable import variable
from itertools import repeat
import collections.abc
from functools import partial
import random
import numpy as np
import math


class BEiT(Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        Module.name = 'BEiT'
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.init_std = init_std

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = initializer_((1, 1, embed_dim), ['truncated_normal', init_std])
        self.mask_token = initializer_((1, 1, embed_dim), ['truncated_normal', init_std])
        if use_abs_pos_emb:
            self.pos_embed = initializer_((1, num_patches + 1, embed_dim), ['truncated_normal', init_std])
        else:
            self.pos_embed = None
        self.pos_drop = dropout(drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = tf.linspace(0., drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = [
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)]
        self.norm = norm_layer(embed_dim)

        self.lm_head = dense(vocab_size, embed_dim)

        Module.apply(self.init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.assign(tf.math.divide(param, math.sqrt(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)
    
    def init_weights(self, l):
        if isinstance(l, dense):
            l.weight.assign(initializer(l.weight.shape, ['truncated_normal', self.init_std]))
        elif isinstance(l, conv2d):
            l.weight.assign(initializer(l.weight.shape, ['truncated_normal', self.init_std]))

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = tf.tile(self.mask_token, (batch_size, seq_len, 1))

        # replace the masked visual tokens by mask_token
        w = tf.cast(tf.expand_dims(bool_masked_pos, axis=-1), mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = tf.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        self.flag=flag
        if flag==0:
            self.param_=self.param.copy()
            self.lm_head_=self.head
            self.lm_head=dense(classes,self.embed_dim)
            param.extend(self.lm_head.param)
            self.param=param
        elif flag==1:
            del self.param_[-len(self.lm_head.param):]
            self.param_.extend(self.lm_head.param)
            self.param=self.param_
        else:
            self.lm_head,self.lm_head_=self.lm_head_,self.lm_head
            del self.param_[-len(self.lm_head.param):]
            self.param_.extend(self.lm_head.param)
            self.param=self.param_
        return

    def __call__(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        if return_all_tokens:
            return self.lm_head(x)
        else:
            return self.lm_head(tf.boolean_mask(x, bool_masked_pos, axis=1))


def beit_base_patch16_224_8k_vocab(**kwargs):
    model = BEiT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layer_norm, epsilon=1e-6), vocab_size=8192, **kwargs)
    return model


def beit_large_patch16_224_8k_vocab(**kwargs):
    model = BEiT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layer_norm, epsilon=1e-6), vocab_size=8192, **kwargs)
    return model


class Mlp:
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = dense(hidden_features, in_features)
        self.act = act_layer
        self.fc2 = dense(out_features, hidden_features)
        self.drop = dropout(drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention:
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = dense(all_head_dim * 3, dim, use_bias=False)
        if qkv_bias:
            self.q_bias = initializer_((all_head_dim), 'zeros')
            self.v_bias = initializer_((all_head_dim), 'zeros')
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = initializer_(
                            (self.num_relative_distance, num_heads), 'zeros')  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = tf.arange(window_size[0])
            coords_w = tf.arange(window_size[1])
            coords = tf.stack(tf.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = tf.reshape(coords, [0, -1])  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = tf.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                tf.zeros((window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = tf.reduce_sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = dropout(attn_drop)
        self.proj = dense(dim, all_head_dim)
        self.proj_drop = dropout(proj_drop)

    def __call__(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = tf.concat((self.q_bias, tf.zeros_like(self.v_bias), self.v_bias), axis=0)
            qkv = tf.matmul(x, self.qkv.weight) + qkv_bias
        else:
            qkv = tf.matmul(x, self.qkv.weight)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = tf.transpose(tf.reshape(qkv, (B, N, 3, self.num_heads, -1)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1))), (
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1))  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = tf.nn.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block:

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=tf.nn.gelu, norm_layer=layer_norm,
                 window_size=None, attn_head_dim=None):
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = stochastic_depth(drop_path) if drop_path > 0. else identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = variable(init_values * tf.ones((dim)))
            self.gamma_2 = variable(init_values * tf.ones((dim)))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def __call__(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed:
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = conv2d(embed_dim, input_size=in_chans, kernel_size=patch_size, strides=patch_size)

    def __call__(self, x, **kwargs):
        B, H, W, C = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = tf.reshape(self.proj(x), (B, -1, self.embed_dim))
        return x


class RelativePositionBias:

    def __init__(self, window_size, num_heads):
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = initializer_(
                        (self.num_relative_distance, num_heads), 'zeros')  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = tf.arange(window_size[0])
        coords_w = tf.arange(window_size[1])
        coords = tf.stack(tf.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = tf.reshape(coords, [0, -1])  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = tf.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            tf.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = tf.reduce_sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def __call__(self):
        relative_position_bias = \
            tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1))), (
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1))  # Wh*Ww,Wh*Ww,nH
        return tf.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww


class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
