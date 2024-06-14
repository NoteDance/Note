""" Bottleneck Self Attention (Bottleneck Transformers)

Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605

@misc{2101.11605,
Author = {Aravind Srinivas and Tsung-Yi Lin and Niki Parmar and Jonathon Shlens and Pieter Abbeel and Ashish Vaswani},
Title = {Bottleneck Transformers for Visual Recognition},
Year = {2021},
}

Based on ref gist at: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

This impl is a WIP but given that it is based on the ref gist likely not too far off.

Hacked together by / Copyright 2024 NoteDance
"""
from typing import List

import tensorflow as tf
from Note import nn


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, heads, height, width, dim)
        rel_k: (2 * width - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    x = tf.matmul(q, rel_k, transpose_b=True)
    x = tf.reshape(x, (-1, W, 2 * W - 1))

    # pad to shift from relative to absolute indexing
    x_pad = tf.pad(x, paddings=[[0, 0], [0, 0], [0, 1]])
    x_pad = tf.reshape(x_pad, [x_pad.shape[0], -1])
    x_pad = tf.pad(x_pad, paddings=[[0, 0], [0, W - 1]])

    # reshape and slice out the padded elements
    x_pad = tf.reshape(x_pad, (-1, W + 1, 2 * W - 1))
    x = x_pad[:, :W, W - 1:]

    # reshape and tile
    x = tf.reshape(x, (B, H, 1, W, W))
    x = tf.tile(x, [1, 1, H, 1, 1])
    return tf.transpose(x, permute_mask)


class PosEmbedRel:
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925
    """
    def __init__(self, feat_size, dim_head, scale):
        self.height, self.width = nn.to_2tuple(feat_size)
        self.dim_head = dim_head
        self.height_rel = nn.Parameter(tf.random.normal((self.height * 2 - 1, dim_head)) * scale)
        self.width_rel = nn.Parameter(tf.random.normal((self.width * 2 - 1, dim_head)) * scale)

    def __call__(self, q):
        B, HW, _ = q.shape

        # relative logits in width dimension.
        q = tf.reshape(q, (B, self.height, self.width, -1))
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = tf.transpose(q, (0, 2, 1, 3))
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = tf.reshape(rel_logits, (B, HW, HW))
        return rel_logits


class BottleneckAttn:
    """ Bottleneck Attention
    Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605

    The internal dimensions of the attention module are controlled by the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query and key (qk) dimensions are determined by
        * num_heads * dim_head if dim_head is not None
        * num_heads * (dim_out * attn_ratio // num_heads) if dim_head is None
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not used

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        stride (int): output stride of the module, avg pool used if stride == 2 (default: 1).
        num_heads (int): parallel attention heads (default: 4)
        dim_head (int): dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool): add bias to q, k, and v projections
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """
    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, dim_head=None,
            qk_ratio=1.0, qkv_bias=False, scale_pos_embed=False):
        assert feat_size is not None, 'A concrete feature size matching expected input (H, W) is required'
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or nn.make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed

        self.qkv = nn.conv2d(self.dim_out_qk * 2 + self.dim_out_v, 1, dim, use_bias=qkv_bias)

        # NOTE I'm only supporting relative pos embedding for now
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head_qk, scale=self.scale)

        self.pool = nn.avg_pool2d(2, 2) if stride == 2 else nn.identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)  # fan-in
        nn.trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        nn.trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def __call__(self, x):
        B, H, W, C = x.shape

        x = self.qkv(x)  # B, H, W, (2 * dim_head_qk + dim_head_v) * num_heads

        # NOTE head vs channel split ordering in qkv projection was decided before I allowed qk to differ from v
        # So, this is more verbose than if heads were before qkv splits, but throughput is not impacted.
        q, k, v = tf.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], axis=-1)
        q = tf.transpose(q, (0, 3, 1, 2))
        k = tf.transpose(k, (0, 3, 1, 2))
        v = tf.transpose(v, (0, 3, 1, 2))
        q = tf.transpose(tf.reshape(q, (B * self.num_heads, self.dim_head_qk, -1)), 
                         (0, 2, 1))
        k = tf.reshape(k, (B * self.num_heads, self.dim_head_qk, -1))  # no transpose, for tf.matmul(q, k)
        v = tf.transpose(tf.reshape(v, (B * self.num_heads, self.dim_head_v, -1)), 
                         (0, 2, 1))

        if self.scale_pos_embed:
            attn = (tf.matmul(q, k) + self.pos_embed(q)) * self.scale  # B * num_heads, H * W, H * W
        else:
            attn = tf.matmul(q, k) * self.scale + self.pos_embed(q)
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.reshape(tf.matmul(attn, v), (B, H, W, self.dim_out_v))  # B, H, W, dim_out
        out = self.pool(out)
        return out
