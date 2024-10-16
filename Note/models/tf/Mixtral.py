from dataclasses import dataclass
from typing import Dict, Optional, Union

import tensorflow as tf
from Note import nn


@dataclass
class ModelArgs:
    model_type: str
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_experts_per_tok: int = 2
    num_key_value_heads: int = 8
    num_local_experts: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class MixtralAttention:
    def __init__(self, args: ModelArgs):
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.rope_theta = args.rope_theta

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.dense(
            self.num_heads * self.head_dim, self.hidden_size, use_bias=False
        )
        self.k_proj = nn.dense(
            self.num_key_value_heads * self.head_dim, self.hidden_size, use_bias=False
        )
        self.v_proj = nn.dense(
            self.num_key_value_heads * self.head_dim, self.hidden_size, use_bias=False
        )
        self.o_proj = nn.dense(
            self.hidden_size, self.num_heads * self.head_dim, use_bias=False
        )

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x,
        mask = None,
        cache = None,
    ):
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = tf.transpose(tf.reshape(queries, (B, L, self.num_heads, -1)), (0, 2, 1, 3))
        keys = tf.transpose(tf.reshape(keys, (B, L, self.num_key_value_heads, -1)), (0, 2, 1, 3))
        values = tf.transpose(tf.reshape(values, (B, L, self.num_key_value_heads, -1)), (
            0, 2, 1, 3
        ))

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = nn.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = tf.reshape(tf.transpose(output, (0, 2, 1, 3)), (B, L, -1))
        return self.o_proj(output)


class MixtralSparseMoeBlock:
    def __init__(self, args: ModelArgs):
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        # gating
        self.gate = nn.dense(self.num_experts, self.hidden_dim, use_bias=False)

        self.switch_mlp = nn.SwitchGLU(self.hidden_dim, self.ffn_dim, self.num_experts)

    def __call__(self, x):
        gates = self.gate(x)

        k = self.num_experts_per_tok
        values, inds = tf.math.top_k(tf.negative(gates), k=k - 1)
        inds = tf.stop_gradient(inds)
        scores = tf.keras.ops.take_along_axis(gates, inds, axis=-1)
        scores = tf.nn.softmax(scores, axis=-1)

        y = self.switch_mlp(x, inds)
        y = tf.reduce_sum((y * scores[..., None]), axis=-2)

        return y


class MixtralDecoderLayer:
    def __init__(self, args: ModelArgs):
        self.hidden_size = args.hidden_size

        self.self_attn = MixtralAttention(args)

        self.block_sparse_moe = MixtralSparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x,
        mask = None,
        cache = None,
    ):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        out = h + r
        return out


class MixtralModel:
    def __init__(self, args: ModelArgs):
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.embedding(args.hidden_size, args.vocab_size)
        self.layers = [
            MixtralDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.create_additive_causal_mask(T)
            mask = tf.cast(mask, h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Mixtral(nn.Model):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = MixtralModel(args)
        self.head = self.dense(args.vocab_size, args.hidden_size, use_bias=False)
        self.args = args

    def __call__(
        self,
        inputs,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return self.head(out)

    def sanitize(self, weights):
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    to_join = [
                        weights.pop(f"{prefix}.block_sparse_moe.experts.{e}.{n}.{k}")
                        for e in range(self.args.num_local_experts)
                    ]
                    if to_join:
                        weights[f"{prefix}.block_sparse_moe.switch_mlp.{m}.{k}"] = (
                            tf.stack(to_join)
                        )
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads