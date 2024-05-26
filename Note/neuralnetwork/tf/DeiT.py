# Copyright (c) NoteDance
# All rights reserved.

import tensorflow as tf
from Note import nn
from itertools import repeat
from typing import Optional
import collections.abc
from functools import partial

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

class Attention:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
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
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
        attn = tf.nn.softmax(attn)
        attn = self.attn_drop(attn)
        x = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.variable(init_values * tf.ones((dim)))
        self.gamma_2 = nn.variable(init_values * tf.ones((dim)))

    def __call__(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_Block_paralx2:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.variable(init_values * tf.ones((dim)))
        self.gamma_1_1 = nn.variable(init_values * tf.ones((dim)))
        self.gamma_2 = nn.variable(init_values * tf.ones((dim)))
        self.gamma_2_1 = nn.variable(init_values * tf.ones((dim)))
        
    def __call__(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
        
class Block_paralx2:
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem:
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768):
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Layers()
        self.proj.add(nn.conv2d(embed_dim//4, 4, in_chans, 4))
        self.proj.add(nn.batch_norm(embed_dim//4, synchronized=True))
        self.proj.add(tf.nn.gelu)
        self.proj.add(nn.conv2d(embed_dim//4, 2, embed_dim//4, 2))
        self.proj.add(nn.batch_norm(embed_dim//4, synchronized=True))
        self.proj.add(tf.nn.gelu)
        self.proj.add(nn.conv2d(embed_dim, 2, embed_dim//4, 2))
        self.proj.add(nn.batch_norm(embed_dim, synchronized=True))

    def __call__(self, x):
        B, C, H, W = x.shape
        x = tf.reshape(self.proj(x), [B, C, H * W])
        x = tf.transpose(x, [0, 2, 1])
        return x
    
class vit_models(nn.Model):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.layer_norm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=tf.nn.gelu,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,**kwargs):
        
        super().__init__()
        nn.Model.add()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.initializer_((1, 1, embed_dim), ['truncated_normal', .02], name='cls_token')

        self.pos_embed = nn.initializer_((1, num_patches, embed_dim), ['truncated_normal', .02], name='pos_embed')

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = [
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)]
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.dense(num_classes, embed_dim) if num_classes > 0 else nn.identity()

        nn.Model.apply(self.init_weights)
        self.training = True

    def init_weights(self, l):
        if isinstance(l, nn.dense):
            l.weight.assign(nn.initializer(l.weight.shape, ['truncated_normal', .02]))
            if l.use_bias==True:
                l.bias.assign(nn.initializer(l.bias.shape, ['truncated_normal', .02]))

    def no_weight_decay(self):
        return ['pos_embed', 'cls_token']

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.dense(num_classes, self.embed_dim) if num_classes > 0 else nn.identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = tf.tile(self.cls_token, (B, 1, 1))
        
        x = x + self.pos_embed
        
        x = tf.concat((cls_tokens, x), axis=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        self.flag=flag
        if flag==0:
            self.param_=self.param.copy()
            self.head_=self.head
            self.head=nn.dense(classes,self.embed_dim)
            param.extend(self.head.param)
            self.param=param
        elif flag==1:
            del self.param_[-len(self.head.param):]
            self.param_.extend(self.head.param)
            self.param=self.param_
        else:
            self.head,self.head_=self.head_,self.head
            del self.param_[-len(self.head.param):]
            self.param_.extend(self.head.param)
            self.param=self.param_
        return

    def __call__(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = nn.dropout(float(self.dropout_rate))(x, self.training)
        x = self.head(x)
        
        return x

# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model
    
def deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model

def deit_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False, **kwargs):
    model = vit_models(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model 

def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model
    
def deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model
    
def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model
    
def deit_huge_patch14_52_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=52, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model
    
def deit_huge_patch14_26x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=26, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Layer_scale_init_Block_paralx2, **kwargs)
    return model
    
def deit_Giant_48x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Block_paral_LS, **kwargs)
    return model

def deit_giant_40x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Block_paral_LS, **kwargs)
    return model

def deit_Giant_48_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model

def deit_giant_40_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model

# Models from Three things everyone should know about Vision Transformers (https://arxiv.org/pdf/2203.09795.pdf)

def deit_small_patch16_36_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model
    
def deit_small_patch16_36(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model
    
def deit_small_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block_paralx2, **kwargs)
    return model
    
def deit_small_patch16_18x2(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Block_paralx2, **kwargs)
    return model
    
  
def deit_base_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block_paralx2, **kwargs)
    return model


def deit_base_patch16_18x2(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Block_paralx2, **kwargs)
    return model
    

def deit_base_patch16_36x1_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model

def deit_base_patch16_36x1(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.layer_norm, epsilon=1e-6), **kwargs)
    return model