import tensorflow as tf
from Note.nn.layer.llama import RoPE
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.embedding import embedding
from Note.nn.Module import Module
import math
from dataclasses import dataclass


@dataclass
class ModelArgs:
    n_positions: int = 2048
    vocab_size: int = 51200
    n_embd: int = 2560
    n_head: int = 32
    n_layer: int = 32
    rotary_dim: int = 32


class RoPEAttention:
    def __init__(self, dims: int, n_head: int, rotary_dim: int):
        self.n_head = n_head

        self.q_proj = dense(dims, dims)
        self.k_proj = dense(dims, dims)
        self.v_proj = dense(dims, dims)
        self.dense = dense(dims, dims)

        self.rope = RoPE(rotary_dim, traditional=False)

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        n_head = self.n_head
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = tf.transpose(tf.reshape(queries, (B, L, n_head, -1)), (0, 2, 1, 3))
        keys = tf.transpose(tf.reshape(keys, (B, L, n_head, -1)), (0, 2, 1, 3))
        values = tf.transpose(tf.reshape(values, (B, L, n_head, -1)), (0, 2, 1, 3))

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = tf.concat([key_cache, keys], axis=2)
            values = tf.concat([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = tf.cast(queries, tf.float32)
        keys = tf.cast(keys, tf.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = tf.matmul((queries * scale), tf.transpose(keys, (0, 1, 3, 2)))
        if mask is not None:
            scores = scores + mask

        scores = tf.cast(tf.nn.softmax(scores, axis=-1), values.dtype)
        values_hat = tf.reshape(tf.transpose(tf.matmul(scores, values), (0, 2, 1, 3)), (B, L, -1))

        return self.dense(values_hat), (keys, values)


class MLP:
    def __init__(self, dim, hidden_dim):
        self.fc1 = dense(hidden_dim, dim)
        self.fc2 = dense(dim, hidden_dim)

    def __call__(self, x):
        return self.fc2(tf.nn.gelu(self.fc1(x), approximate="precise"))


class ParallelBlock:
    def __init__(self, config: ModelArgs):
        dims = config.n_embd
        mlp_dims = dims * 4
        self.self_attn = RoPEAttention(dims, config.n_head, config.rotary_dim)
        self.input_layernorm = layer_norm(dims)
        self.mlp = MLP(dims, mlp_dims)

    def __call__(self, x, mask, cache):
        h = self.input_layernorm(x)
        attn_h, cache = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x, cache


class Transformer:
    def __init__(self, config: ModelArgs):
        self.embed_tokens = embedding(config.n_embd, config.vocab_size)
        self.layers = [ParallelBlock(config) for i in range(config.n_layer)]
        self.final_layernorm = layer_norm(config.n_embd)

    def __call__(self, x, mask, cache):
        x = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        return self.final_layernorm(x), cache


class Phi2:
    def __init__(self, config: ModelArgs):
        Module.init()
        self.model = Transformer(config)
        self.lm_head = dense(config.vocab_size, config.n_embd)
        self.param = Module.param
    
    def fine_tuning(self,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.lm_head_=self.lm_head
            self.lm_head=dense(ModelArgs.vocab_size, ModelArgs.n_embd)
            param.extend(self.lm_head.param)
            self.param=param
        elif flag==1:
            del self.param_[-len(self.lm_head.param):]
            self.param_.extend(self.lm_head.param)
            self.param=self.param_
        else:
            self.lm_head,self.lm_head_=self.lm_head_,self.lm_head
            del self.param_[-len(self.lm_head.param):]
            self.param_.extend(self.lm_head.param)
            self.param=self.param_
        return

    def __call__(
        self,
        x,
        mask = None,
        cache = None,
    ):
        mask = None
        if x.shape[1] > 1:
            mask = tf.fill((x.shape[1], x.shape[1]), float("-inf"))
            mask = tf.linalg.band_part(mask, 0, -1)
            mask = tf.linalg.set_diag(mask, tf.zeros(x.shape[1]))
            mask = tf.cast(mask, x.dtype)

        y, cache = self.model(x, mask, cache)
        return self.lm_head(y), cache