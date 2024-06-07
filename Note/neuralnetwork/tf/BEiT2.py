# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By NoteDance
# --------------------------------------------------------'

import tensorflow as tf
from Note import nn
from functools import partial
from itertools import repeat
import collections.abc
import math


class VisionTransformerForMaskedImageModeling(nn.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        nn.Model.add()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_heads = num_heads

        self.cls_token = nn.initializer_((1, 1, embed_dim), ['truncated_normal', init_std], name='cls_token')
        self.mask_token = nn.initializer_((1, 1, embed_dim), ['truncated_normal', init_std])
        if use_abs_pos_emb:
            self.pos_embed = nn.initializer_((1, num_patches + 1, embed_dim), ['truncated_normal', init_std], name='pos_embed')
        else:
            self.pos_embed = None
        self.pos_drop = nn.dropout(drop_rate)

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

        self.init_std = init_std
        self.lm_head = nn.dense(vocab_size, embed_dim)

        self.lm_head.weight.assign(nn.initializer_(self.lm_head.weight.shape, ['truncated_normal', self.init_std]))
        nn.Model.apply(self.init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.assign(tf.math.divide(param, math.sqrt(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)
    
    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', self.init_std]))
        elif isinstance(l, nn.conv2d):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', self.init_std]))

    def no_weight_decay(self):
        return ['pos_embed', 'cls_token']

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = tf.tile(self.mask_token, (batch_size, seq_len, 1))

        # replace the masked visual tokens by mask_token
        w = tf.cast(tf.expand_dims(bool_masked_pos, -1), mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = tf.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def __call__(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = tf.zeros((x.shape[0], self.patch_embed.num_patches), dtype=tf.bool)
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = tf.zeros((x.shape[0], self.patch_embed.num_patches), dtype=tf.bool)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = tf.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = tf.split(x, 3, axis=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = tf.transpose(tf.reshape(q, (b, n, self.num_heads, -1)), (0, 2, 1, 3))
            k = tf.transpose(tf.reshape(k, (b, n, self.num_heads, -1)), (0, 2, 1, 3))
            v = tf.transpose(tf.reshape(v, (b, n, self.num_heads, -1)), (0, 2, 1, 3))
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v


    def forward_intermediate(self, x, bool_masked_pos=None, layer_id=12):
        if bool_masked_pos is None:
            bool_masked_pos = tf.zeros((x.shape[0], self.patch_embed.num_patches), dtype=tf.bool)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = tf.cast(tf.expand_dims(bool_masked_pos, -1), mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = tf.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = tf.transpose(tf.reshape(patch_pos_embed, [1, int(math.sqrt(N)), int(math.sqrt(N)), dim]), (0, 3, 1, 2))
        new_width = w0 / math.sqrt(N)
        new_height = h0 / math.sqrt(N)
        patch_pos_embed = tf.image.resize(patch_pos_embed, [int(new_height), int(new_width)], method='bicubic')
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = tf.reshape(tf.transpose(patch_pos_embed, (0, 2, 3, 1)), (1, -1, dim))
        return tf.concat((tf.expand_dims(class_pos_embed, 0), patch_pos_embed), axis=1)

    def get_last_selfattention(self, x):
        B, nc, w, h = x.shape

        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))
        x = tf.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                x = x + self.interpolate_pos_encoding(x, w, h)
            else:
                x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)
                
class VisionTransformerForMaskedImageModelingCLS(VisionTransformerForMaskedImageModeling):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02,
                 early_layers=6, head_layers=2, shared_lm_head=True):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias, init_std=init_std)

        self.early_layers = early_layers
        print(f'early layer {early_layers}, late layer {depth - early_layers}, condenser head layers {head_layers}, shared_lm_head {shared_lm_head}')

        dpr = tf.linspace(0., drop_path_rate, max(depth, early_layers + head_layers))  # stochastic depth decay rule
        self.cls_pt_layers = [
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(early_layers, early_layers + head_layers)]
        self.fix_init_cls_pt_weight()

        self.shared_lm_head = shared_lm_head
        if not shared_lm_head:
            self.cls_pt_norm = norm_layer(embed_dim)
            self.cls_pt_lm_head = nn.dense(vocab_size, embed_dim, weight_initializer=['truncated_normal', self.init_std])

    def fix_init_cls_pt_weight(self):
        def rescale(param, layer_id):
            param.assign(tf.math.divide(param, math.sqrt(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.cls_pt_layers):
            rescale(layer.attn.proj.weight, self.early_layers + layer_id + 1)
            rescale(layer.mlp.fc2.weight, self.early_layers + layer_id + 1)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = tf.tile(self.mask_token, (batch_size, seq_len, 1))

        # replace the masked visual tokens by mask_token
        w = tf.cast(tf.expand_dims(bool_masked_pos, -1), mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = tf.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i + 1 == self.early_layers:
                early_states = x[:, 1:]

        x_cls_pt = tf.concat([x[:, 0:1], early_states], axis=1)
        for blk in self.cls_pt_layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=rel_pos_bias)

        return self.norm(x), self.norm(x_cls_pt) if self.shared_lm_head else self.cls_pt_norm(x_cls_pt)

    def __call__(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = tf.zeros((x.shape[0], self.patch_embed.num_patches), dtype=tf.bool)
        x, x_cls_pt = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        if return_patch_tokens:
            return [x, x_cls_pt]
        if return_all_tokens:
            return [self.lm_head(x), self.lm_head(x_cls_pt) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt)]
        else:
            # return the masked tokens
            return [self.lm_head(x[bool_masked_pos]), self.lm_head(x_cls_pt[bool_masked_pos]) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt[bool_masked_pos])]
    

def beit_base_patch16_224_8k_vocab_cls_pt(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_base_patch16_224_8k_vocab(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_base_patch16_192_8k_vocab(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=192, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_base_patch16_256_8k_vocab(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_24x544_patch16_224_8k_vocab(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=224, patch_size=16, embed_dim=544, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_24x544_patch16_224_8k_vocab_cls_pt(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        img_size=224, patch_size=16, embed_dim=544, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_large_patch16_224_8k_vocab(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_large_patch16_224_8k_vocab_cls_pt(**kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model

def beit_huge_patch14_224_8k_vocab(**kwargs):
    # patch_size=14, embed_dim=1280, depth=32, num_heads=16
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), vocab_size=vocab_size, **kwargs)
    return model


class Mlp:
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.dense(hidden_features, in_features)
        self.act = act_layer
        self.fc2 = nn.dense(out_features, hidden_features)
        self.drop = nn.dropout(drop)

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

        self.qkv = nn.dense(all_head_dim * 3, dim, use_bias=False)
        if qkv_bias:
            self.q_bias = nn.initializer_((all_head_dim), 'zeros')
            self.v_bias = nn.initializer_((all_head_dim), 'zeros')
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.initializer_(
                            (self.num_relative_distance, num_heads), 'zeros')  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = tf.range(window_size[0])
            coords_w = tf.range(window_size[1])
            coords = tf.stack(tf.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
            coords_flatten = tf.reshape(coords, [coords.shape[0], -1])  # 2, Wh*Ww
            self.relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            self.relative_coords = tf.Variable(tf.transpose(self.relative_coords, (1, 2, 0)))  # Wh*Ww, Wh*Ww, 2
            self.relative_coords[:, :, 0].assign(self.relative_coords[:, :, 0] + (window_size[0] - 1))  # shift to start from 0
            self.relative_coords[:, :, 1].assign(self.relative_coords[:, :, 1] + (window_size[1] - 1))
            self.relative_coords[:, :, 0].assign(self.relative_coords[:, :, 0] * (2 * window_size[1] - 1))
            self.relative_position_index = \
                    tf.Variable(tf.zeros((window_size[0] * window_size[1] + 1, ) * 2, dtype=self.relative_coords.dtype))
            self.relative_position_index[1:, 1:].assign(tf.reduce_sum(self.relative_coords, axis=-1))  # Wh*Ww, Wh*Ww
            self.relative_position_index[0, 0:].assign(self.num_relative_distance - 3)
            self.relative_position_index[0:, 0].assign(self.num_relative_distance - 2)
            self.relative_position_index[0, 0].assign(self.num_relative_distance - 1)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, all_head_dim)
        self.proj_drop = nn.dropout(proj_drop)

    def __call__(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = tf.concat((self.q_bias, tf.zeros_like(self.v_bias), self.v_bias), axis=0)
            qkv = tf.matmul(x, self.qkv.weight) + qkv_bias
        else:
            qkv = tf.matmul(x, self.qkv.weight)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = tf.transpose(tf.reshape(qkv, (B, N, 3, self.num_heads, -1)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)

        q = q * self.scale
        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1))), (
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1))  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
            attn = attn + tf.expand_dims(relative_position_bias, 0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = tf.nn.softmax(attn)
        attn = self.attn_drop(attn)
        
        if return_attention:
            return attn

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_qkv:
            return x, qkv
        
        return x


class Block:

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=tf.nn.gelu, norm_layer=nn.layer_norm,
                 window_size=None, attn_head_dim=None):
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than nn.dropout here
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * tf.ones((dim)))
            self.gamma_2 = nn.Parameter(init_values * tf.ones((dim)))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def __call__(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv
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

        self.proj = nn.conv2d(embed_dim, input_size=in_chans, kernel_size=patch_size, strides=patch_size)

    def __call__(self, x, **kwargs):
        B, H, W, C = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = tf.reshape(self.proj(x), (B, -1, self.embed_dim))
        return x


class RelativePositionBias:

    def __init__(self, window_size, num_heads):
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.initializer_(
                        (self.num_relative_distance, num_heads), 'zeros')  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = tf.arange(window_size[0])
        coords_w = tf.arange(window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = tf.reshape(coords, [0, -1])  # 2, Wh*Ww
        self.relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        self.relative_coords = tf.Variable(tf.transpose(self.relative_coords, (1, 2, 0)))  # Wh*Ww, Wh*Ww, 2
        self.relative_coords[:, :, 0].assign(self.relative_coords[:, :, 0] + (window_size[0] - 1))  # shift to start from 0
        self.relative_coords[:, :, 1].assign(self.relative_coords[:, :, 1] + (window_size[1] - 1))
        self.relative_coords[:, :, 0].assign(self.relative_coords[:, :, 0] * (2 * window_size[1] - 1))
        self.relative_position_index = \
            tf.Variable(tf.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=self.relative_coords.dtype))
        self.relative_position_index[1:, 1:].assign(tf.reduce_sum(self.relative_coords, axis=-1))  # Wh*Ww, Wh*Ww
        self.relative_position_index[0, 0:].assign(self.num_relative_distance - 3)
        self.relative_position_index[0:, 0].assign(self.num_relative_distance - 2)
        self.relative_position_index[0, 0].assign(self.num_relative_distance - 1)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def __call__(self):
        relative_position_bias = \
            tf.reshape(tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1))), (
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1))  # Wh*Ww,Wh*Ww,nH
        return tf.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww