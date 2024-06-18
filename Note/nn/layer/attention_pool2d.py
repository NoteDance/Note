""" Attention Pool 2D

Implementations of 2D spatial feature pooling using multi-head attention instead of average pool.

Based on idea in CLIP by OpenAI, licensed Apache 2.0
https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

Hacked together by / Copyright 2024 NoteDance
"""
from typing import Optional, Union, Tuple

import tensorflow as tf
from Note import nn


class RotAttentionPool2d:
    """ Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at differeing resolutions from
    train varies widely and falls off dramatically. I'm not sure if there is a way around this... -RW
    """
    def __init__(
            self,
            in_features: int,
            out_features: Optional[int] = None,
            ref_feat_size: Union[int, Tuple[int, int]] = 7,
            embed_dim: Optional[int] = None,
            head_dim: Optional[int] = 64,
            num_heads: Optional[int] = None,
            qkv_bias: bool = True,
            qkv_separate: bool = False,
            pool_type: str = 'token',
            class_token: bool = False,
            drop_rate: float = 0.,
            use_fused_attn = True
    ):
        assert pool_type in ('', 'token')
        self.embed_dim = embed_dim = embed_dim or in_features
        self.in_features = in_features
        self.out_features = out_features or in_features
        ref_feat_size = nn.to_2tuple(ref_feat_size)
        if num_heads is not None:
            assert embed_dim % num_heads == 0
            head_dim = embed_dim // num_heads
        else:
            assert embed_dim % head_dim == 0
            num_heads = embed_dim // head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool_type = pool_type.lower()
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn

        if class_token:
            self.cls_token = nn.Parameter(tf.zeros((1, embed_dim)))
        else:
            self.cls_token = None

        if qkv_separate:
            self.q = nn.dense(embed_dim, in_features, use_bias=qkv_bias)
            self.k = nn.dense(embed_dim, in_features, use_bias=qkv_bias)
            self.v = nn.dense(embed_dim, in_features, use_bias=qkv_bias)
            self.qkv = None
        else:
            self.qkv = nn.dense(embed_dim * 3, in_features, use_bias=qkv_bias)
        self.drop = nn.dropout(drop_rate)
        self.proj = nn.dense(self.out_features, embed_dim)
        self.pos_embed = nn.RotaryEmbedding(self.head_dim, in_pixels=False, ref_feat_shape=ref_feat_size)

    def init_weights(self, zero_init_last: bool = False):
        if self.qkv is None:
            in_features = self.q.in_features
            nn.trunc_normal_(self.q.weight, std=in_features ** -0.5)
            nn.trunc_normal_(self.k.weight, std=in_features ** -0.5)
            nn.trunc_normal_(self.v.weight, std=in_features ** -0.5)
        else:
            in_features = self.qkv.in_features
            nn.trunc_normal_(self.qkv.weight, std=in_features ** -0.5)

    def reset(self, num_classes: Optional[int] = None, pool_type: Optional[str] = None):
        # NOTE: this module is being used as a head, so need compatible reset()
        if pool_type is not None:
            assert pool_type in ('', 'token')
            self.pool_type = pool_type
        if num_classes is not None:
            self.proj = nn.dense(num_classes, self.in_features) if num_classes > 0 else nn.identity()
            self.out_features = num_classes if num_classes > 0 else self.embed_dim

    def _pool(self, x, H: int, W: int):
        if self.pool_type == 'token':
            x = x[:, 0]
        else:
            # if not pooled, return spatial output without token
            x = tf.reshape(x[:, 1:], (x.shape[0], H, W, -1))
        return x

    def __call__(self, x, pre_logits: bool = False):
        B, H, W, _ = x.shape
        N = H * W
        x = tf.reshape(x, (B, N, -1))
        if self.cls_token is None:
            x = tf.concat([tf.reduce_mean(x, axis=1, keepdims=True), x], axis=1)
        else:
            x = tf.concat([tf.tile(self.cls_token, (x.shape[0], 1, 1)), x], axis=1)
        if self.qkv is None:
            q = tf.transpose(tf.reshape(self.q(x), (B, N + 1, self.num_heads, self.head_dim)), 
                             (0, 2, 1, 3))
            k = tf.transpose(tf.reshape(self.k(x), (B, N + 1, self.num_heads, self.head_dim)), 
                             (0, 2, 1, 3))
            v = tf.transpose(tf.reshape(self.v(x), (B, N + 1, self.num_heads, self.head_dim)), 
                             (0, 2, 1, 3))
        else:
            x = tf.transpose(tf.reshape(self.qkv(x), (B, N + 1, 3, self.num_heads, self.head_dim)), 
                             (2, 0, 3, 1, 4))
            q, k, v = tf.unstack(x, axis=0)

        rse, rce = self.pos_embed.get_embed((H, W))
        q = tf.cast(tf.concat([q[:, :, :1, :], nn.apply_rot_embed(q[:, :, 1:, :], rse, rce)], axis=2), v.dtype)
        k = tf.cast(tf.concat([k[:, :, :1, :], nn.apply_rot_embed(k[:, :, 1:, :], rse, rce)], axis=2), v.dtype)

        if self.fused_attn:
            x = nn.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
            attn = tf.nn.softmax(attn, axis=-1)
            x = tf.matmul(attn, v)
        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, N + 1, -1))
        x = self.drop(x)
        if pre_logits:
            x = self._pool(x, H, W)
            return x
        x = self.proj(x)
        x = self._pool(x, H, W)
        return x


class AttentionPool2d:
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """
    def __init__(
            self,
            in_features: int,
            feat_size: Union[int, Tuple[int, int]] = 7,
            out_features: Optional[int] = None,
            embed_dim: Optional[int] = None,
            head_dim: Optional[int] = 64,
            num_heads: Optional[int] = None,
            qkv_bias: bool = True,
            qkv_separate: bool = False,
            pool_type: str = 'token',
            class_token: bool = False,
            drop_rate: float = 0.,
            use_fused_attn = True
    ):
        assert pool_type in ('', 'token')
        self.embed_dim = embed_dim = embed_dim or in_features
        self.in_features = in_features
        self.out_features = out_features or in_features
        if num_heads is not None:
            assert embed_dim % num_heads == 0
            head_dim = embed_dim // num_heads
        else:
            assert embed_dim % head_dim == 0
            num_heads = embed_dim // head_dim
        self.feat_size = nn.to_2tuple(feat_size)
        self.seq_len = self.feat_size[0] * self.feat_size[1]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool_type = pool_type
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn

        if class_token:
            self.cls_token = nn.Parameter(tf.zeros((1, embed_dim)))
        else:
            self.cls_token = None

        if qkv_separate:
            self.q = nn.dense(embed_dim, in_features, use_bias=qkv_bias)
            self.k = nn.Linear(embed_dim, in_features, use_bias=qkv_bias)
            self.v = nn.Linear(embed_dim, in_features, use_bias=qkv_bias)
            self.qkv = None
        else:
            self.q = self.k = self.v = None
            self.qkv = nn.dense(embed_dim * 3, in_features, use_bias=qkv_bias)
        self.drop = nn.dropout(drop_rate)
        self.proj = nn.dense(self.out_features, embed_dim)
        self.pos_embed = nn.Parameter(tf.zeros((self.seq_len + 1, in_features)))

        self.init_weights()

    def init_weights(self, zero_init_last: bool = False):
        if self.qkv is None:
            in_features = self.q.in_features
            nn.trunc_normal_(self.q.weight, std=in_features ** -0.5)
            nn.trunc_normal_(self.k.weight, std=in_features ** -0.5)
            nn.trunc_normal_(self.v.weight, std=in_features ** -0.5)
        else:
            in_features = self.qkv.in_features
            nn.trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.trunc_normal_(self.pos_embed, std=in_features ** -0.5)

    def reset(self, num_classes: Optional[int] = None, pool_type: Optional[str] = None):
        # NOTE: this module is being used as a head, so need compatible reset()
        if pool_type is not None:
            assert pool_type in ('', 'token')
            self.pool_type = pool_type
        if num_classes is not None:
            self.proj = nn.dense(num_classes, self.in_features) if num_classes > 0 else nn.identity()
            self.out_features = num_classes if num_classes > 0 else self.embed_dim

    def _pool(self, x, H: int, W: int):
        if self.pool_type == 'token':
            x = x[:, 0]
        else:
            # if not pooled, return spatial output without token
            x = tf.reshape(x[:, 1:], (x.shape[0], H, W, -1))
        return x

    def __call__(self, x, pre_logits: bool = False):
        B, H, W, _ = x.shape
        N = H * W
        x = tf.reshape(x, (B, N, -1))
        if self.cls_token is None:
            x = tf.concat([tf.reduce_mean(x, axis=1, keepdims=True), x], axis=1)
        else:
            x = tf.concat([tf.tile(self.cls_token, (x.shape[0], 1, 1)), x], axis=1)
        pos_embed = nn.resample_abs_pos_embed(tf.expand_dims(self.pos_embed, axis=0), (H, W), num_prefix_tokens=1)
        x = x + pos_embed

        if self.qkv is None:
            q = tf.transpose(tf.reshape(self.q(x), (B, N + 1, self.num_heads, self.head_dim)), 
                             (0, 2, 1, 3))
            k = tf.transpose(tf.reshape(self.k(x), (B, N + 1, self.num_heads, self.head_dim)), 
                             (0, 2, 1, 3))
            v = tf.transpose(tf.reshape(self.v(x), (B, N + 1, self.num_heads, self.head_dim)), 
                             (0, 2, 1, 3))
        else:
            x = tf.transpose(tf.reshape(self.qkv(x), (B, -1, 3, self.num_heads, self.head_dim)), 
                             (2, 0, 3, 1, 4))
            q, k, v = tf.unstack(x, axis=0)

        if self.fused_attn:
            x = nn.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
            attn = tf.nn.softmax(attn, axis=-1)
            x = tf.matmul(attn, v)
        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, N + 1, -1))
        x = self.drop(x)
        if pre_logits:
            x = self._pool(x, H, W)
            return x
        x = self.proj(x)
        x = self._pool(x, H, W)
        return x