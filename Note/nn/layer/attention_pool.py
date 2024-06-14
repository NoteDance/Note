from typing import Optional

import tensorflow as tf
from Note import nn


class AttentionPoolLatent:
    """ Attention pooling w/ latent query
    """

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            feat_size: Optional[int] = None,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = 'token',
            norm_layer = None,
            drop: float = 0.0,
            use_fused_attn = True
    ):
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn

        if pos_embed == 'abs':
            assert feat_size is not None
            self.pos_embed = nn.Parameter(tf.zeros((feat_size, in_features)))
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(tf.zeros((1, self.latent_len, embed_dim)))

        self.q = nn.dense(embed_dim, embed_dim, use_bias=qkv_bias)
        self.kv = nn.dense(embed_dim * 2, embed_dim, use_bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.identity()
        self.proj = nn.dense(embed_dim, embed_dim)
        self.proj_drop = nn.dropout(drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.identity()
        self.mlp = nn.Mlp(embed_dim, int(embed_dim * mlp_ratio))

        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            nn.trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        nn.trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def __call__(self, x):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + tf.cast(tf.expand_dims(self.pos_embed, axis=0), x.dtype)

        q_latent = tf.tile(self.latent, (B, 1, 1))
        q = tf.transpose(tf.reshape(self.q(q_latent), (B, self.latent_len, self.num_heads, self.head_dim)), 
                         (0, 2, 1, 3))

        kv = tf.transpose(tf.reshape(self.kv(x), (B, N, 2, self.num_heads, self.head_dim)), 
                          (2, 0, 3, 1, 4))
        k, v = tf.unstack(kv, axis=0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = nn.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
            attn = tf.nn.softmax(attn, axis=-1)
            x = tf.matmul(attn, v)
        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, self.latent_len, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = tf.reduce_mean(x, axis=1)
        return x