# Copyright NoteDance All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Modifed from Note. https://github.com/NoteDance/Note/tree/Note-7.0/Note/nn/layer/vision_transformer.py

"""

import tensorflow as tf
from Note import nn
import numpy as np
import math
from functools import partial


class PatchEmbed:
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        img_size = nn.to_2tuple(img_size)
        patch_size = nn.to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Layers()
                self.proj.add(nn.conv2d(embed_dim // 4, 7, in_chans, strides=4, padding=3))
                self.proj.add(tf.nn.relu)
                self.proj.add(nn.Conv2d(embed_dim // 2, 3, embed_dim // 4, strides=3, padding=0))
                self.proj.add(tf.nn.relu)
                self.proj.add(nn.Conv2d(embed_dim, 3, embed_dim // 2, strides=1, padding=1))
            elif patch_size[0] == 16:
                self.proj = nn.Layers()
                self.proj.add(nn.conv2d(embed_dim // 4, 7, in_chans, strides=4, padding=3))
                self.proj.add(tf.nn.relu)
                self.proj.add(nn.Conv2d(embed_dim // 2, 3, embed_dim // 4, strides=2, padding=1))
                self.proj.add(tf.nn.relu)
                self.proj.add(nn.Conv2d(embed_dim, 3, embed_dim // 2, strides=2, padding=1))
        else:
            self.proj = nn.conv2d(embed_dim, patch_size, in_chans, strides=patch_size)

    def __call__(self, x):
        B, H, W, C = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, (B, H*W, C))
        return x


class CrossAttention:
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.dense(dim, dim, use_bias=qkv_bias)
        self.wk = nn.dense(dim, dim, use_bias=qkv_bias)
        self.wv = nn.dense(dim, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

    def __call__(self, x):

        B, N, C = x.shape
        q = tf.transpose(tf.reshape(self.wq(x[:, 0:1, ...]), (B, 1, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))  # B1C -> B1H(C/H) -> BH1(C/H)
        k = tf.transpose(tf.reshape(self.wk(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H)
        v = tf.transpose(tf.reshape(self.wv(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, 1, C))   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock:

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm, has_mlp=True):
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def __call__(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock:

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm):
        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = []
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    nn.Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          proj_drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                layers = nn.Layers()
                layers.add(tmp)
                self.blocks.append(layers)

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = []
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer, nn.dense(dim[(d+1) % num_branches], dim[d])]
            layers = nn.Layers()
            layers.add(tmp)
            self.projs.append(layers)

        self.fusion = []
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                layers = nn.Layers()
                layers.add(tmp)
                self.fusion.append(layers)

        self.revert_projs = []
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer, nn.dense(dim[d], dim[(d+1) % num_branches])]
            layers = nn.Layers()
            layers.add(tmp)
            self.revert_projs.append(layers)

    def __call__(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = tf.concat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), axis=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = tf.concat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), axis=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]


class CrossViT(nn.Model):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.layer_norm, multi_conv=False):
        super().__init__()
        nn.Model.add()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = nn.to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)

        self.patch_embed = []
        if hybrid_backbone is None:
            self.pos_embed = [nn.Parameter(tf.zeros((1, 1 + num_patches[i], embed_dim[i]))) for i in range(self.num_branches)]
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(tf.Variable(get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), trainable=False))

            del self.pos_embed
            self.pos_embed = [nn.Parameter(tf.zeros((1, 1 + num_patches[i], embed_dim[i]))) for i in range(self.num_branches)]

        self.cls_token = [nn.Parameter(tf.zeros((1, 1, embed_dim[i]))) for i in range(self.num_branches)]
        self.pos_drop = nn.dropout(drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x for x in tf.linspace(0., drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = []
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = [norm_layer(embed_dim[i]) for i in range(self.num_branches)]
        self.head = [nn.dense(num_classes, embed_dim[i]) if num_classes > 0 else nn.identity() for i in range(self.num_branches)]

        for i in range(self.num_branches):
            if self.pos_embed[i].trainable:
                self.pos_embed[i].assign(nn.initializer(self.pos_embed[i].shape, ['truncated_normal', .02]))
            self.cls_token[i].assign(nn.initializer(self.cls_token[i].shape, ['truncated_normal', .02]))

        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))

    def no_weight_decay(self):
        out = ['cls_token']
        if self.pos_embed[0].trainable:
            out.append('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.dense(num_classes, self.embed_dim) if num_classes > 0 else nn.identity()

    def forward_features(self, x):
        B, H, W, C = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = nn.interpolate(x, (self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = tf.tile(self.cls_token[i], (B, 1, 1))  # stole cls_tokens impl from Phil Wang, thanks
            tmp = tf.concat((cls_tokens, tmp), axis=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]

        return out

    def __call__(self, x):
        xs = self.forward_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = tf.reduce_mean(tf.stack(ce_logits, axis=0), axis=0)
        return ce_logits




def crossvit_tiny_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model


def crossvit_small_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model


def crossvit_base_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[12, 12], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model


def crossvit_9_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model


def crossvit_15_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model


def crossvit_18_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model


def crossvit_9_dagger_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), multi_conv=True, **kwargs)
    return model

def crossvit_15_dagger_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), multi_conv=True, **kwargs)
    return model

def crossvit_15_dagger_384(**kwargs):
    model = CrossViT(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), multi_conv=True, **kwargs)
    return model

def crossvit_18_dagger_224(**kwargs):
    model = CrossViT(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), multi_conv=True, **kwargs)
    return model

def crossvit_18_dagger_384(**kwargs):
    model = CrossViT(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.layer_norm, epsilon=1e-6), multi_conv=True, **kwargs)
    return model


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return tf.expand_dims(tf.convert_to_tensor(sinusoid_table, 'float32'), 0)


class Token_performer:
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
    # def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.0, dp2=0.0):
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.dense(3 * self.emb, dim)
        self.dp = nn.dropout(dp1)
        self.proj = nn.dense(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.layer_norm(dim)
        self.norm2 = nn.layer_norm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Layers()
        self.mlp.add(nn.dense(1 * self.emb, self.emb))
        self.mlp.add(tf.nn.gelu)
        self.mlp.add(nn.dense(self.emb, 1 * self.emb))
        self.mlp.add(nn.dropout(dp2))

        self.m = int(self.emb * kernel_ratio)
        self.w = tf.random.normal([self.m, self.emb])
        self.w = tf.Variable(tf.linalg.orthogonalize(self.w) * math.sqrt(self.m), trainable=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = tf.repeat((tf.reduce_sum((x * x), axis=-1, keepdims=True)), repeats=self.m, axis=-1) / 2
        wtx = tf.einsum('bti,mi->btm', tf.cast(x, 'float32'), self.w)

        return tf.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = tf.split(self.kqv(x), self.emb, axis=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = tf.expand_dims(tf.einsum('bti,bi->bt', qp, tf.reduce_sum(kp, axis=1)), axis=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = tf.einsum('bin,bim->bnm', tf.cast(v, 'float32'), kp)  # (B, emb, m)
        y = tf.einsum('bti,bni->btn', qp, kptv) / (tf.repeat(D, repeats=self.emb, axis=-1) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def __call__(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x