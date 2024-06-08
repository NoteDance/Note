import tensorflow as tf
from Note import nn
from typing import Optional


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
        self.patch_size = nn.to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = nn.to_2tuple(img_size)
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


class Attention:
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.layer_norm, use_fused_attn=True):
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.dense(dim * 3, dim, use_bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.identity()
        self.attn_drop = nn.dropout(attn_drop)
        self.proj = nn.dense(dim, dim)
        self.proj_drop = nn.dropout(proj_drop)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            x = nn.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.rate if nn.Model.training else 0.,
            )
        else:
            q = q * self.scale
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)))
            attn = tf.nn.softmax(attn)
            attn = self.attn_drop(attn)
            x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale:
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
    ):
        self.gamma = nn.variable(init_values * tf.ones(dim))

    def __call__(self, x):
        return x * self.gamma
    
    
class Block:
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, proj_drop=0., attn_drop=0., init_values=None,
                 drop_path=0., act_layer=tf.nn.gelu, norm_layer=nn.layer_norm, mlp_layer=nn.Mlp
                 ):
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.identity()
        self.drop_path1 = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.identity()
        self.drop_path2 = nn.stochastic_depth(drop_path) if drop_path > 0. else nn.identity()

    def __call__(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
