from typing import Optional

import tensorflow as tf
from Note import nn


class MultiQueryAttentionV2:
    """Multi Query Attention.

    Fast Transformer Decoding: One Write-Head is All You Need
    https://arxiv.org/pdf/1911.02150.pdf

    This is an acceletor optimized version - removing multiple unneccessary
    tensor transpose by re-arranging indices according to the following rules: 1)
    contracted indices are at the end, 2) other indices have the same order in the
    input and output tensores.

    Compared to V1, this gives 3x speed up.
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 8,
            key_dim: int = 64,
            value_dim: int = 64,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        """Initializer."""
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = key_dim ** -0.5

        self.query_proj = nn.Parameter(tf.random.normal([self.num_heads, self.key_dim, dim]))
        self.key_proj = nn.Parameter(tf.random.normal([dim, self.key_dim]))
        self.value_proj = nn.Parameter(tf.random.normal([dim, self.value_dim]))
        self.attn_drop = nn.dropout(attn_drop)
        self.out_proj = nn.Parameter(tf.random.normal([dim_out, self.num_heads, self.value_dim]))
        self.proj_drop = nn.dropout(proj_drop)

    def _reshape_input(self, t):
        """Reshapes a tensor to three dimensions, keeping the first and last."""
        s = t.shape
        # Propagate the shape statically where possible.
        #num = t.shape[1:-1].numel()
        #return t.reshape(s[0], num, s[-1])
        return tf.reshape(t, (s[0], -1, s[-1]))

    def __call__(self, x, m = None):
        """Run layer computation."""
        s = x.shape
        m = m or x

        reshaped_x = self._reshape_input(x)
        reshaped_m = self._reshape_input(m)

        q = tf.einsum('bnd,hkd->bnhk', reshaped_x, self.query_proj)
        k = tf.einsum('bmd,dk->bmk', reshaped_m, self.key_proj)

        attn = tf.einsum('bnhk,bmk->bnhm', q, k)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        v = tf.einsum('bmd,dv->bmv', reshaped_m, self.value_proj)
        o = tf.einsum('bnhm,bmv->bnhv', attn, v)
        result = tf.einsum('bnhv,dhv->bnd', o, self.out_proj)
        result = self.proj_drop(result)
        return tf.reshape(result, s)


class Attention2d:

    """ multi-head attention for 2D NHWC tensors"""
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 32,
            bias: bool = True,
            expand_first: bool = False,
            head_first: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            use_fused_attn = True
    ):
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.head_first = head_first
        self.scale = num_heads ** -0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.conv2d(dim_attn * 3, 1, dim, use_bias=bias)
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.conv2d(dim_out, 1, dim, use_bias=bias)
        self.proj_drop = nn.dropout(proj_drop)

    def __call__(self, x, attn_mask = None):
        B, H, W, C = x.shape

        if self.head_first:
            q, k, v = tf.split(tf.reshape(self.qkv(x), (B, -1, self.num_heads, self.dim_head * 3)), 3, axis=-1)
        else:
            q, k, v = tf.unstack(tf.reshape(self.qkv(x), (B, -1, 3, self.num_heads, self.dim_head)), axis=2)

        if self.fused_attn:
            x = nn.scaled_dot_product_attention(
                tf.transpose(q, (0, 2, 1, 3)),
                tf.transpose(k, (0, 2, 1, 3)),
                tf.transpose(v, (0, 2, 1, 3)),
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.rate if self.training else 0.,
            )
            x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, H, W, -1))
        else:
            q = q * self.scale
            attn = tf.matmul(tf.transpose(q, (0, 2, 1, 3)), k)
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = tf.nn.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = tf.reshape(tf.matmul(v, tf.transpose(attn, (0, 2, 1, 3))), (B, H, W, -1))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x