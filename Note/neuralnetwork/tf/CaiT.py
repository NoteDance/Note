# Copyright (c) 2024-present, NoteDance, Inc.
# All rights reserved.

import tensorflow as tf
from Note import nn
from itertools import repeat
from typing import Optional
import collections.abc
from functools import partial


__all__ = [
    'cait_M48', 'cait_M36',
    'cait_S36', 'cait_S24','cait_S24_224',
    'cait_XS24','cait_XXS24','cait_XXS24_224',
    'cait_XXS36','cait_XXS36_224'
]


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class PatchEmbed:
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            flatten: bool = True,
            bias: bool = True,
    ):
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten

        self.proj = nn.conv2d(embed_dim, input_size=in_chans, kernel_size=patch_size, strides=patch_size, use_bias=bias)

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        if self.flatten:
            x = tf.reshape(x, [B, H*W, C])  # NHWC -> NLC
        return x


class Mlp:
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=tf.nn.gelu,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.dense(hidden_features, in_features, use_bias=bias[0])
        self.act = act_layer
        self.drop1 = nn.dropout(drop_probs[0])
        self.fc2 = nn.dense(out_features, hidden_features, use_bias=bias[1])
        self.drop2 = nn.dropout(drop_probs[1])

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x, approximate="tanh")
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x  


class Class_Attention:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.dense(dim, dim, use_bias=qkv_bias)
        self.k = nn.dense(dim, dim, use_bias=qkv_bias)
        self.v = nn.dense(dim, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

    
    def __call__(self, x ):
        
        B, N, C = x.shape
        q = tf.transpose(tf.reshape(tf.expand_dims(self.q(x[:,0]), 1), (B, 1, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))
        k = tf.transpose(tf.reshape(self.k(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))

        q = q * self.scale
        v = tf.transpose(tf.reshape(self.v(x), (B, N, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))

        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_cls = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls     
        
class LayerScale_Block_CA:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4):
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * tf.ones((dim)))
        self.gamma_2 = nn.Parameter(init_values * tf.ones((dim)))

    
    def __call__(self, x, x_cls):
        
        u = tf.concat((x_cls,x),axis=1)
        
        
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls 
        
        
class Attention_talking_head:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.attn_drop = nn.dropout(attn_drop)
        
        self.proj = nn.dense(dim, dim)
        
        self.proj_l = nn.dense(num_heads, num_heads)
        self.proj_w = nn.dense(num_heads, num_heads)
        
        self.proj_drop = nn.dropout(proj_drop)


    
    def __call__(self, x):
        B, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
    
        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) 
        
        attn = tf.transpose(self.proj_l(tf.transpose(attn, (0,2,3,1))), (0,3,1,2))
                
        attn = tf.nn.softmax(attn, axis=-1)
  
        attn = tf.transpose(self.proj_w(tf.transpose(attn, (0,2,3,1))), (0,3,1,2))
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale_Block:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm,Attention_block = Attention_talking_head,
                 Mlp_block=Mlp,init_values=1e-4):
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * tf.ones((dim)))
        self.gamma_2 = nn.Parameter(init_values * tf.ones((dim)))

    def __call__(self, x):        
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 
    
    
class CaiT(nn.Model):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.layer_norm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA,
                 Patch_layer=PatchEmbed,act_layer=tf.nn.gelu,
                 Attention_block = Attention_talking_head,Mlp_block=Mlp,
                init_scale=1e-4,
                Attention_block_token_only=Class_Attention,
                Mlp_block_token_only= Mlp, 
                depth_token_only=2,
                mlp_ratio_clstk = 4.0):
        super().__init__()
        nn.Model.add()

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.initializer((1, 1, embed_dim), ['truncated_normal', .02], name='cls_token')
        self.pos_embed = nn.initializer((1, num_patches, embed_dim), ['truncated_normal', .02], name='pos_embed')
        self.pos_drop = nn.dropout(drop_rate)

        dpr = [drop_path_rate for i in range(depth)] 
        self.blocks = [
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)]
        

        self.blocks_token_only = [
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)]
            
        self.norm = norm_layer(embed_dim)


        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = self.dense(num_classes, embed_dim) if num_classes > 0 else nn.identity()

        nn.Model.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.trunc_normal_(l.weight, std=.02))

    def no_weight_decay(self):
        return ['pos_embed', 'cls_token']

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = tf.tile(self.cls_token, (B, 1, 1))
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x,cls_tokens)

        x = tf.concat((cls_tokens, x), axis=1)
            
                
        x = self.norm(x)
        return x[:, 0]

    def __call__(self, x):
        x = self.forward_features(x)
        
        x = self.head(x)

        return x 
        
    
def cait_XXS24_224(**kwargs):
    model = CaiT(
        img_size= 224,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    return model 

def cait_XXS24(**kwargs):
    model = CaiT(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    return model 

def cait_XXS36_224(**kwargs):
    model = CaiT(
        img_size= 224,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    return model 

def cait_XXS36(**kwargs):
    model = CaiT(
        img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    return model 



def cait_XS24(**kwargs):
    model = CaiT(
        img_size= 384,patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    return model 



def cait_S24_224(**kwargs):
    model = CaiT(
        img_size= 224,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    return model 

def cait_S24(**kwargs):
    model = CaiT(
        img_size= 384,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    return model 

def cait_S36(**kwargs):
    model = CaiT(
        img_size= 384,patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    return model 



def cait_M36(**kwargs):
    model = CaiT(
        img_size= 384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    return model 

def cait_M48(**kwargs):
    model = CaiT(
        img_size= 448 , patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    return model       