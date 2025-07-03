""" Lambda Layer

Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
    - https://arxiv.org/abs/2102.08602

@misc{2102.08602,
Author = {Irwan Bello},
Title = {LambdaNetworks: Modeling Long-Range Interactions Without Attention},
Year = {2021},
}

Status:
This impl is a WIP. Code snippets in the paper were used as reference but
good chance some details are missing/wrong.

I've only implemented local lambda conv based pos embeddings.

For a PyTorch impl that includes other embedding options checkout
https://github.com/lucidrains/lambda-networks

Hacked together by / Copyright 2024 NoteDance
"""
import tensorflow as tf
from Note import nn


def ndgrid(*tensors):
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    """
    try:
        return tf.meshgrid(*tensors, indexing='ij')
    except TypeError:
        return tf.meshgrid(*tensors)
    

def rel_pos_indices(size):
    size = nn.to_2tuple(size)
    stacked = tf.stack(ndgrid(tf.keras.ops.arange(size[0]), tf.keras.ops.arange(size[1])))
    pos = tf.reshape(stacked, (stacked.shape[0], -1))
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos  # 2, H * W, H * W


class LambdaLayer(nn.Layer):
    """Lambda Layer

    Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
        - https://arxiv.org/abs/2102.08602

    NOTE: intra-depth parameter 'u' is fixed at 1. It did not appear worth the complexity to add.

    The internal dimensions of the lambda module are controlled via the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query (q) and key (k) dimension are determined by
        * dim_head = (dim_out * attn_ratio // num_heads) if dim_head is None
        * q = num_heads * dim_head, k = dim_head
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not set

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map for relative pos variant H, W
        stride (int): output stride of the module, avg pool used if stride == 2
        num_heads (int): parallel attention heads.
        dim_head (int): dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        r (int): local lambda convolution radius. Use lambda conv if set, else relative pos if not. (default: 9)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool): add bias to q, k, and v projections
    """
    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, dim_head=16, r=9,
            qk_ratio=1.0, qkv_bias=False):
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, ' should be divided by num_heads'
        super().__init__()
        self.dim_qk = dim_head or nn.make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.num_heads = num_heads
        self.dim_v = dim_out // num_heads

        self.qkv = nn.conv2d(
            num_heads * self.dim_qk + self.dim_qk + self.dim_v,
            kernel_size=1, input_size=dim, use_bias=qkv_bias)
        self.norm_q = nn.batch_norm(num_heads * self.dim_qk)
        self.norm_v = nn.batch_norm(self.dim_v)

        if r is not None:
            # local lambda convolution for pos
            self.conv_lambda = nn.conv3d(self.dim_qk, (r, r, 1), 1, padding=(r // 2, r // 2, 0))
            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            # relative pos embedding
            assert feat_size is not None
            feat_size = nn.to_2tuple(feat_size)
            rel_size = [2 * s - 1 for s in feat_size]
            self.conv_lambda = None
            self.pos_emb = nn.Parameter(tf.zeros((rel_size[0], rel_size[1], self.dim_qk)))
            self.register_buffer('rel_pos_indices', rel_pos_indices(feat_size), persistent=False)

        self.pool = nn.avg_pool2d(2, 2) if stride == 2 else nn.identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)  # fan-in
        if self.conv_lambda is not None:
            nn.trunc_normal_(self.conv_lambda.weight, std=self.dim_qk ** -0.5)
        if self.pos_emb is not None:
            nn.trunc_normal_(self.pos_emb, std=.02)

    def __call__(self, x):
        B, H, W, C = x.shape
        M = H * W
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, [
            self.num_heads * self.dim_qk, self.dim_qk, self.dim_v], axis=-1)
        q = tf.transpose(tf.reshape(self.norm_q(q), (B, M, self.num_heads, self.dim_qk)), (0, 2, 1, 3))  # B, num_heads, M, K
        v = tf.reshape(self.norm_v(v), (B, M, self.dim_v))  # B, M, V
        k = tf.nn.softmax(tf.transpose(tf.reshape(k, (B, M, self.dim_qk)), (0, 2, 1)), axis=-1)  # B, K, M

        content_lam = tf.matmul(k, v)  # B, K, V
        content_out = tf.matmul(q, tf.expand_dims(content_lam, axis=1))  # B, num_heads, M, V

        if self.pos_emb is None:
            position_lam = self.conv_lambda(tf.reshape(v, (B, 1, H, W, self.dim_v)))  # B, H, W, V, K
            position_lam = tf.reshape(position_lam, (B, 1, H * W, self.dim_qk, self.dim_v))  # B, 1, M, K, V
        else:
            # FIXME relative pos embedding path not fully verified
            pos_emb = tf.tile(self.pos_emb[self.rel_pos_indices[0], self.rel_pos_indices[1]], (B, 1, 1, 1))
            position_lam = tf.expand_dims(tf.matmul(tf.transpose(pos_emb, (0, 1, 3, 2)), tf.expand_dims(v, axis=1)), axis=1)  # B, 1, M, K, V
        position_out = tf.squeeze(tf.matmul(tf.expand_dims(q, axis=-2), position_lam), axis=-2)  # B, num_heads, M, V

        out = tf.reshape(tf.transpose((content_out + position_out), (0, 2, 1, 3)), (B, H, W, C))  # B, H, W, C (num_heads * V)
        out = self.pool(out)
        return out