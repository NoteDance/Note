""" Halo Self Attention

Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
    - https://arxiv.org/abs/2103.12731

@misc{2103.12731,
Author = {Ashish Vaswani and Prajit Ramachandran and Aravind Srinivas and Niki Parmar and Blake Hechtman and
    Jonathon Shlens},
Title = {Scaling Local Self-Attention for Parameter Efficient Visual Backbones},
Year = {2021},
}

Status:
This impl is a WIP, there is no official ref impl and some details in paper weren't clear to me.
The attention mechanism works but it's slow as implemented.

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


class HaloAttn:
    """ Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731

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
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """
    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, num_heads=8, dim_head=None, block_size=8, halo_size=3,
            qk_ratio=1.0, qkv_bias=False, avg_down=False, scale_pos_embed=False, is_xla=True):
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or nn.make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.is_xla = is_xla
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.block_stride = 1
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_avg_pool else stride
            self.block_size_ds = self.block_size // self.block_stride

        # FIXME not clear if this stride behaviour is what the paper intended
        # Also, the paper mentions using a 3D conv for dealing with the blocking/gather, and leaving
        # data in unfolded block form. I haven't wrapped my head around how that'd look.
        self.q = nn.conv2d(self.dim_out_qk, 1, dim, strides=self.block_stride, use_bias=qkv_bias)
        self.kv = nn.conv2d(self.dim_out_qk + self.dim_out_v, 1, dim, use_bias=qkv_bias)

        self.pos_embed = PosEmbedRel(
            block_size=self.block_size_ds, win_size=self.win_size, dim_head=self.dim_head_qk, scale=self.scale)

        self.pool = nn.avg_pool2d(2, 2) if use_avg_pool else nn.identity()

        self.reset_parameters()

    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5  # fan-in
        nn.trunc_normal_(self.q.weight, std=std)
        nn.trunc_normal_(self.kv.weight, std=std)
        nn.trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        nn.trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def __call__(self, x):
        B, H, W, C = x.shape
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.q(x)
        # unfold
        q = tf.transpose(tf.reshape(q, (
            -1,num_h_blocks, self.block_size_ds, num_w_blocks, 
            self.block_size_ds, self.dim_head_qk,)), (0, 1, 2, 4, 3, 5))
        # B, num_blocks, num_heads * dim_head * block_size ** 2
        q = tf.reshape(q, (B * self.num_heads, num_blocks, -1, self.dim_head_qk))
        # B * num_heads, num_blocks, block_size ** 2, dim_head

        kv = self.kv(x)
        # Generate overlapping windows for kv. This approach is good for GPU and CPU. However, unfold() is not
        # lowered for PyTorch XLA so it will be very slow. See code at bottom of file for XLA friendly approach.
        # FIXME figure out how to switch impl between this and conv2d if XLA being used.
        if self.is_xla:
            WW = self.win_size ** 2
            pw = tf.reshape(tf.eye(WW, dtype=x.dtype), (self.win_size, self.win_size, 1, WW))
            kv = tf.reshape(kv, (-1, H, W, 1))
            kv_pad = nn.zeropadding2d(padding=self.halo_size)(kv)
            kv = tf.nn.conv2d(kv_pad, pw, strides=self.block_size)
        else:
            paddings = [[0, 0], [self.halo_size, self.halo_size], [self.halo_size, self.halo_size], [0, 0]]
            kv = tf.pad(kv, paddings, mode='CONSTANT')
            kv = nn.unfold(self.win_size, self.block_size)(kv)
            kv = tf.transpose(tf.reshape(kv, (B * self.num_heads, 
                                    self.dim_head_qk + self.dim_head_v, num_blocks, -1)), 
                                    (0, 2, 3, 1))
        k, v = tf.split(kv, [self.dim_head_qk, self.dim_head_v], axis=-1)
        # B * num_heads, num_blocks, win_size ** 2, dim_head_qk or dim_head_v

        if self.scale_pos_embed:
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2)) + self.pos_embed(q)) * self.scale
        else:
            attn = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scale + self.pos_embed(q)
        # B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v).transpose(1, 3)  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        # fold
        out = tf.reshape(out, (-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks))
        out = tf.reshape(tf.transpose(out, (0, 3, 1, 4, 2)), (
            B, self.dim_out_v, H // self.block_stride, W // self.block_stride))
        out = tf.transpose(out, (0, 2, 3, 1))
        # B, H // block_stride, W // block_stride, dim_out
        out = self.pool(out)
        return out