# Copyright (c) NoteDance
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import tensorflow as tf
from Note import nn

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = nn.initializer_((dim), 'ones')

    def _norm(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.pow(x, 2), -1, keepdims=True) + self.eps) 

    def __call__(self, x):
        output = tf.cast(self._norm(tf.cast(x, 'float32')), x.dtype)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (tf.cast(tf.range(0, dim, 2)[: (dim // 2)], 'float32') / dim))
    t = tf.range(end, dtype='float32')
    freqs = tf.experimental.numpy.outer(t, freqs)
    freqs_cis = tf.complex(tf.ones_like(freqs), freqs)
    real_part = tf.math.cos(freqs)
    imag_part = tf.math.sin(freqs)
    freqs_cis = tf.complex(real_part, imag_part)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return tf.reshape(freqs_cis, shape)


def apply_rotary_emb(
    xq,
    xk,
    freqs_cis,
):
    xq = tf.reshape(tf.cast(xq, 'float32'), (xq.shape[:-1] + (xq.shape[-1] // 2, 2)))
    real_part = xq[..., 0]
    imag_part = xq[..., 1]
    xq_ = tf.complex(real_part, imag_part)
    xk = tf.reshape(tf.cast(xk, 'float32'), (xk.shape[:-1] + (xk.shape[-1] // 2, 2)))
    real_part = xk[..., 0]
    imag_part = xk[..., 1]
    xk_ = tf.complex(real_part, imag_part)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_freqs_cis = xq_ * freqs_cis
    xq_out = tf.stack([tf.math.real(xq_freqs_cis), tf.math.imag(xq_freqs_cis)], axis=-1)
    shape = xq_out.shape
    xq_out = tf.reshape(xq_out, [-1, shape[1], shape[2], shape[3] * shape[4]])
    xk_freqs_cis = xk_ * freqs_cis
    xk_out = tf.stack([tf.math.real(xk_freqs_cis), tf.math.imag(xk_freqs_cis)], axis=-1)
    shape = xk_out.shape
    xk_out = tf.reshape(xk_out, [-1, shape[1], shape[2], shape[3] * shape[4]])
    return tf.cast(xq_out, xq.dtype), tf.cast(xk_out, xk.dtype)


def repeat_kv(x, n_rep: int):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return tf.reshape(tf.tile(x[:, :, :, None, :], [1, 1, 1, n_rep, 1]), (bs, slen, n_kv_heads * n_rep, head_dim))


class Attention:
    def __init__(self, args: ModelArgs):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.dense(
            args.n_heads * self.head_dim,
            args.dim,
            use_bias=False,
        )
        self.wk = nn.dense(
            self.n_kv_heads * self.head_dim,
            args.dim,
            use_bias=False,
        )
        self.wv = nn.dense(
            self.n_kv_heads * self.head_dim,
            args.dim,
            use_bias=False,
        )
        self.wo = nn.dense(
            args.dim,
            args.n_heads * self.head_dim,
            use_bias=False,
        )

        self.cache_k = tf.Variable(tf.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ))
        self.cache_v = tf.Variable(tf.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ))

    def __call__(
        self,
        x,
        start_pos: int,
        freqs_cis,
        mask,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = tf.reshape(xq, (bsz, seqlen, self.n_local_heads, self.head_dim))
        xk = tf.reshape(xk, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        xv = tf.reshape(xv, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = tf.cast(self.cache_k, xq.dtype)
        self.cache_v = tf.cast(self.cache_v, xq.dtype)

        self.cache_k[:bsz, start_pos : start_pos + seqlen].assign(xk)
        self.cache_v[:bsz, start_pos : start_pos + seqlen].assign(xv)

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = tf.transpose(xq, (0, 2, 1, 3))  # (bs, n_local_heads, seqlen, head_dim)
        keys = tf.transpose(keys, (0, 2, 1, 3))  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = tf.transpose(values,
            (0, 2, 1, 3)
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = tf.matmul(xq, tf.transpose(keys, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = tf.cast(tf.nn.softmax(tf.cast(scores, 'float32')), xq.dtype)
        output = tf.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = tf.reshape(tf.transpose(output, (0, 2, 1, 3)), (bsz, seqlen, -1))
        return self.wo(output)


class FeedForward:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.dense(
            hidden_dim, dim, use_bias=False
        )
        self.w2 = nn.dense(
            dim, hidden_dim, use_bias=False
        )
        self.w3 = nn.dense(
            hidden_dim, dim, use_bias=False
        )

    def __call__(self, x):
        return self.w2(tf.nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock:
    def __init__(self, layer_id: int, args: ModelArgs):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(
        self,
        x,
        start_pos,
        freqs_cis,
        mask,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama3(nn.Model):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.embedding(
            params.dim, params.vocab_size
        )

        self.layers = []
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.head = self.dense(
            params.vocab_size, params.dim, use_bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def __call__(self, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = tf.fill((seqlen, seqlen), float("-inf"))
            mask = tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0)
            mask = tf.linalg.set_diag(mask, tf.zeros(seqlen))
            zero_mask = tf.zeros((seqlen, start_pos), dtype=h.dtype)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = tf.concat([zero_mask, mask], axis=1)
            mask = tf.cast(mask, h.dtype)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = tf.cast(self.head(h), 'float32')
        return output