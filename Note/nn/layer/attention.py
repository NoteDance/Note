from typing import Optional

import tensorflow as tf
from Note import nn


def maybe_add_mask(scores: tf.Tensor, attn_mask: Optional[tf.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask


class Attention(nn.Layer):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer = None,
            use_fused_attn = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.identity()
        self.attn_drop = nn.dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.identity()
        self.proj = nn.dense(dim, dim, use_bias=proj_bias)
        self.proj_drop = nn.dropout(proj_drop)
        
        nn.Model.register(self)

    def __call__(
            self,
            x: tf.Tensor,
            attn_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        B, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv, axis=0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = nn.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.rate if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
            attn = maybe_add_mask(attn, attn_mask)
            attn = tf.nn.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionRope(nn.Layer):
    """ A Self Attention module with ROPE support.

    Includes options for:
     * QK normalization option
     * Attention output (scale) normalization
     * Fused or unfused QKV projection support
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer = None,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            use_fused_attn = None,
    ):
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to add a bias term to the query, key, and value projections
            num_prefix_tokens: Number of reg/cls tokens at the beginning of the sequence that
                should not have position embeddings applied
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for the output projection
            attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            norm_layer: Normalization layer constructor to use for QK and scale normalization
            qk_norm: Enable normalization of query (Q) and key (K) vectors with norm_layer
            scale_norm: Enable normalization (scaling) of attention output with norm_layer
        """
        super().__init__()
        if scale_norm or qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        attn_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn

        if qkv_fused:
            self.qkv = nn.dense(attn_dim * 3, dim, use_bias=qkv_bias)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv = None
            self.q_proj = nn.dense(attn_dim, dim, use_bias=qkv_bias)
            self.k_proj = nn.dense(attn_dim, dim, use_bias=qkv_bias)
            self.v_proj = nn.dense(attn_dim, dim, use_bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(attn_dim) if scale_norm else nn.identity()
        self.proj = nn.dense(dim, attn_dim, use_bias=proj_bias)
        self.proj_drop = nn.dropout(proj_drop)
        nn.Model.register(self)

    def __call__(
            self,
            x,
            rope: Optional[tf.Tensor] = None,
            attn_mask: Optional[tf.Tensor] = None,
    ):
        """Forward pass for the attention module.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            rope: Rotary position embeddings tensor for position-aware attention
            attn_mask: Optional attention mask to apply during attention computation

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = tf.transpose(tf.reshape(qkv, (B, N, 3, self.num_heads, -1)), (2, 0, 3, 1, 4))
            q, k, v = tf.unstack(qkv, axis=0)  # B, num_heads, N, head_dim
        else:
            q = tf.transpose(tf.reshape(self.q_proj(x), (B, N, self.num_heads, -1)), (0, 2, 1, 3))  # B, num_heads, N, C
            k = tf.transpose(tf.reshape(self.k_proj(x), (B, N, self.num_heads, -1)), (0, 2, 1, 3))
            k = tf.transpose(tf.reshape(self.v_proj(x), (B, N, self.num_heads, -1)), (0, 2, 1, 3))

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = tf.cast(tf.concat([q[:, :, :npt, :], nn.apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2), v.dtype)
            k = tf.cast(tf.concat([k[:, :, :npt, :], nn.apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2), v.dtype)

        if self.fused_attn:
            x = nn.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.rate if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
            attn = maybe_add_mask(attn, attn_mask)
            attn = tf.nn.softmax(attn, axis=-1)

            attn = self.attn_drop(attn)
            x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
