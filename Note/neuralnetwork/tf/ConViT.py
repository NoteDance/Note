# Copyright (c) 2024 NoteDance.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from Note import nn
from itertools import repeat
import collections.abc
from functools import partial


class Mlp:
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        nn.Model.add()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.dense(hidden_features, in_features)
        self.act = act_layer
        self.fc2 = nn.dense(out_features, hidden_features)
        self.drop = nn.dropout(drop)
        nn.Model.apply(self.init_weights)
    
    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
            
    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GPSA:
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        nn.Model.add()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.dense(dim * 2, dim, use_bias=qkv_bias)       
        self.v = nn.dense(dim, dim, use_bias=qkv_bias)       
        
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.pos_proj = nn.dense(num_heads, 3)
        self.proj_drop = nn.dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.initializer_((self.num_heads), 'ones')
        nn.Model.apply(self.init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)
    
    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
        
    def __call__(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.shape[1]!=N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = tf.transpose(tf.reshape(self.v(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))
        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape        
        qk = tf.transpose(tf.reshape(self.qk(x), (B, N, 2, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        q, k = qk[0], qk[1]
        pos_score = tf.tile(self.rel_indices, (B, 1, 1,1))
        pos_score = tf.transpose(self.pos_proj(pos_score), (0,3,1,2))
        patch_score = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale
        patch_score = tf.nn.softmax(patch_score)
        pos_score = tf.nn.softmax(pos_score)

        gating = tf.reshape(self.gating_param, (1,-1,1,1))
        attn = (1.-tf.sigmoid(gating)) * patch_score + tf.sigmoid(gating) * pos_score
        attn /= tf.expand_dims(tf.reduce_sum(attn, axis=-1), -1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = tf.reduce_mean(self.get_attention(x), axis=0) # average over batch
        distances = tf.squeeze(self.rel_indices.squeeze)[:,:,-1]**.5
        dist = tf.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.shape[0]
        if return_map:
            return dist, attn_map
        else:
            return dist
    
    def local_init(self, locality_strength=1.):
        
        self.v.weight.assign(tf.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)
        
        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight[2,position].assign(-1)
                self.pos_proj.weight[1,position].assign(2*(h1-center)*locality_distance)
                self.pos_proj.weight[0,position].assign(2*(h2-center)*locality_distance)
        self.pos_proj.weight.assign(self.pos_proj.weight * locality_strength)

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices = tf.Variable(tf.zeros((1, num_patches, num_patches, 3)))
        ind = tf.reshape(tf.range(img_size), (1,-1)) - tf.reshape(tf.range(img_size), (-1, 1))
        indx = tf.tile(ind, [img_size, img_size])
        indy = tf.tile(ind, [img_size, 1])
        indy = tf.tile(indy, [1, img_size])
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2].assign(tf.cast(tf.expand_dims(indd, 0), rel_indices.dtype))
        rel_indices[:,:,:,1].assign(tf.cast(tf.expand_dims(indy, 0), rel_indices.dtype))
        rel_indices[:,:,:,0].assign(tf.cast(tf.expand_dims(indx, 0), rel_indices.dtype))
        self.rel_indices = rel_indices

 
class MHSA:
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        nn.Model.add()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)
        nn.Model.apply(self.init_weights)
    
    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))

    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_map = tf.reduce_mean(tf.nn.softmax(attn_map), 0)

        img_size = int(N**.5)
        ind = tf.reshape(tf.range(img_size), (1,-1)) - tf.reshape(tf.range(img_size), (-1, 1))
        indx = tf.tile(ind, [img_size, img_size])
        indy = tf.tile(ind, [img_size, 1])
        indy = tf.tile(indy, [1, img_size])
        indd = indx**2 + indy**2
        distances = indd**.5

        dist = tf.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N
        
        if return_map:
            return dist, attn_map
        else:
            return dist
            
    def __call__(self, x):
        B, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = tf.nn.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block:

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm, use_gpsa=True, **kwargs):
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
    

class PatchEmbed:
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        nn.Model.add()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.conv2d(embed_dim, input_size=in_chans, kernel_size=patch_size, strides=patch_size)
        nn.Model.apply(self.init_weights)
    
    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
        
    def __call__(self, x):
        B, H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H*W, C])
        return x


class HybridEmbed:
    """ CNN Feature Map Embedding, from timm
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with tf.stop_gradient():
                o = self.backbone(tf.zeros(1, img_size[0], img_size[1], in_chans))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.dense(embed_dim, feature_dim)

    def __call__(self, x):
        x = self.backbone(x)[-1]
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H*W, C])
        x = self.proj(x)
        return x


class VisionTransformer(nn.Model):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=48, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.layer_norm, global_pool=None,
                 local_up_to_layer=10, locality_strength=1., use_pos_embed=True):
        super().__init__()
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        
        self.cls_token = nn.initializer_((1, 1, embed_dim), ['truncated_normal', .02])
        self.pos_drop = nn.dropout(drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.initializer_((1, num_patches, embed_dim), ['truncated_normal', .02])

        dpr = tf.linspace(0., drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = [
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(depth)]
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = self.dense(num_classes, embed_dim) if num_classes > 0 else nn.identity()

        self.head.weight.assign(nn.initializer(self.head.weight.shape, ['truncated_normal', .02]))

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.dense(num_classes, self.embed_dim) if num_classes > 0 else nn.identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = tf.tile(self.cls_token, (B, 1, 1))

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u,blk in enumerate(self.blocks):
            if u == self.local_up_to_layer :
                x = tf.concat((cls_tokens, x), axis=1)
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def __call__(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convit_tiny(**kwargs):
    num_heads = 4
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model

def convit_small(**kwargs):
    num_heads = 9
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model

def convit_base(**kwargs):
    num_heads = 16
    kwargs['embed_dim'] *= num_heads
    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model