# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model implementation."""

import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.initializer import initializer_
from Note.nn.Module import Module
import dataclasses


@dataclasses.dataclass
class GemmaConfig:
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0):
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(tf.cast(tf.range(0, dim, 2)[:(dim // 2)], 'float32') / dim))
    t = tf.range(end)
    freqs = tf.cast(tf.experimental.numpy.outer(t, freqs), 'float32')
    freqs_cis = tf.complex(tf.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """Applies the rotary embedding to the query and key tensors."""
    x_ = tf.complex(
            *tf.split(tf.cast(tf.transpose(x, [0, 2, 1, 3]), 'float32'), num_or_size_splits=2, axis=-1),
                    )
    x_ = x_ * tf.cast(freqs_cis, x_.dtype)
    x_out = tf.cast(tf.stack(tf.math.real(x_),
                             tf.math.imag(x_), axis=-1), x.dtype)
    x_out = tf.concat(tf.split(x_out, num_or_size_splits=2, axis=-1), axis=-2)
    x_out = tf.transpose(tf.reshape(x_out, (x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1)), (0, 2, 1, 3))
    return x_out


class Embedder:
  """Embedder module."""
  def __init__(self, config: GemmaConfig):
    self.vocab_size = config.vocab_size
    self.embed_dim = config.hidden_size
    self.input_embedding_table = initializer_((self.vocab_size, self.embed_dim), 'normal', 'float32')
    self.input_embedding_table_ = tf.transpose(self.input_embedding_table)
    Module.param.append(self.input_embedding_table_)

  def encode(self, x):
    x = tf.gather(self.input_embedding_table, x)
    x *= tf.cast(tf.math.sqrt(self.embed_dim), x.dtype)
    return x

  def decode(self, x):
    return tf.matmul(x, self.input_embedding_table_)


class RMSNorm:

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = initializer_((dim), 'zeros', 'float32')

    def _norm(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.math.pow(x, 2), axis=-1, keepdims=True) + self.eps)

    def __call__(self, x):
        x = tf.cast(self._norm(tf.cast(x, 'float32')), x.dtype)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output


class GemmaMLP:

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.gate_proj = dense(intermediate_size, hidden_size)
        self.up_proj = dense(intermediate_size, hidden_size)
        self.down_proj = dense(hidden_size, intermediate_size)

    def __call__(self, x):
        gate = self.gate_proj(x)
        gate = tf.nn.gelu(gate)
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention:

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = dense(
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            self.hidden_size,
            )
        self.o_proj = dense(
            self.hidden_size,
            self.num_heads * self.head_dim,
            )

    def __call__(
        self,
        hidden_states,
        freqs_cis,
        kv_write_indices,
        kv_cache,
        mask,
    ):
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = tf.split(qkv, [self.q_size, self.kv_size, self.kv_size],
                               axis=-1)

        xq = tf.reshape(xq, (batch_size, -1, self.num_heads, self.head_dim))
        xk = tf.reshape(xk, (batch_size, -1, self.num_kv_heads, self.head_dim))
        xv = tf.reshape(xv, (batch_size, -1, self.num_kv_heads, self.head_dim))

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache = tf.tensor_scatter_nd_update(k_cache, kv_write_indices, xk)
        v_cache = tf.tensor_scatter_nd_update(v_cache, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            batch_size, seq_len, num_heads, head_dim = key.shape
            key = tf.reshape(tf.tile(key[:, :, :, None, :], [1, 1, 1, self.num_queries_per_kv, 1]), 
                       [batch_size, seq_len, num_heads * self.num_queries_per_kv, head_dim])
            batch_size, seq_len, num_heads, head_dim = value.shape
            value = tf.reshape(tf.tile(value[:, :, :, None, :], [1, 1, 1, self.num_queries_per_kv, 1]), 
                       [batch_size, seq_len, num_heads * self.num_queries_per_kv, head_dim])

        # [batch_size, n_local_heads, input_len, head_dim]
        q = tf.transpose(xq, (0, 2, 1, 3))
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = tf.transpose(key, (0, 2, 1, 3))
        v = tf.transpose(value, (0, 2, 1, 3))

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = tf.matmul(q, tf.transpose(k, (0, 1, 3, 2))) * self.scaling
        scores = scores + mask
        scores = tf.cast(tf.nn.softmax(tf.cast(scores, 'float32'), axis=-1), q.dtype)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = tf.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = tf.reshape((tf.transpose(output, (0, 2, 1, 3)),
            (batch_size, input_len, -1)))
        output = self.o_proj(output)
        return output


class GemmaDecoderLayer:

    def __init__(
        self,
        config: GemmaConfig,
    ):
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states,
        freqs_cis,
        kv_write_indices,
        kv_cache,
        mask,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma:

    def __init__(self, config: GemmaConfig, freqs_cis, kv_write_indices, kv_caches, mask):
        Module.init()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.freqs_cis = freqs_cis
        self.kv_write_indices = kv_write_indices
        self.kv_caches = kv_caches
        self.mask = mask

        self.embedder = Embedder()
        self.layers = []
        for _ in range(config.num_hidden_layers):
            self.layers.append(GemmaDecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = dense(config.vocab_size, config.hidden_size)
        
        self.opt=tf.keras.optimizers.Adam()
        self.param = Module.param
    
    def fine_tuning(self,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.embedder_=self.embedder
            self.embedder=Embedder()
            param.extend([self.embedder.input_embedding_table, self.embedder.input_embedding_table_])
            self.param=param
            self.opt.lr=lr
        elif flag==1:
            self.param_.remove(self.embedder_.input_embedding_table)
            self.param_.remove(self.embedder_.input_embedding_table_)
            self.param_.extend([self.embedder.input_embedding_table, self.embedder.input_embedding_table_])
            self.param=self.param_
        else:
            self.embedder,self.embedder_=self.embedder_,self.embedder
            self.param_.remove(self.embedder_.input_embedding_table)
            self.param_.remove(self.embedder_.input_embedding_table_)
            self.param_.extend([self.embedder.input_embedding_table, self.embedder.input_embedding_table_])
            self.param=self.param_
        return

    def fp(
        self,
        data,
    ):
        hidden_states = self.embedder.encode(data)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=self.freqs_cis,
                kv_write_indices=self.kv_write_indices,
                kv_cache=self.kv_caches[i],
                mask=self.mask,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.embedder.decode(hidden_states)
        return logits
    
    def loss(self, output, labels):
        output = tf.reshape(output, [-1, output.shape[-1]])
        labels = tf.reshape(labels, [-1, labels.shape[-1]])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
        return tf.reduce_mean(loss)