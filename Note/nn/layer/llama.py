import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.activation import activation_dict
from Note.nn.initializer import initializer_


class LlamaAttention:
    def __init__(self, dims: int, num_heads: int, dtype='float32'):
        self.num_heads = num_heads
        self.rope = RoPE(dims // num_heads, True)
        self.query_proj = dense(dims, dims, use_bias=False, dtype=dtype)
        self.key_proj = dense(dims, dims, use_bias=False, dtype=dtype)
        self.value_proj = dense(dims, dims, use_bias=False, dtype=dtype)
        self.out_proj = dense(dims, dims, use_bias=False, dtype=dtype)
        self.output_size = self.out_proj.output_size
        self.param = [self.query_proj.param, self.key_proj.param, self.value_proj.param, self.out_proj.param]

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        queries = tf.reshape(queries, (B, L, num_heads, -1))
        queries = tf.transpose(queries, (0, 2, 1, 3))
        keys = tf.reshape(queries, (B, L, num_heads, -1))
        keys = tf.transpose(queries, (0, 2, 1, 3))
        values = tf.reshape(queries, (B, L, num_heads, -1))
        values = tf.transpose(queries, (0, 2, 1, 3))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = tf.concat([key_cache, keys], axis=2)
            values = tf.concat([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = tf.math.sqrt(1 / queries.shape[-1])
        keys = tf.transpose(keys, (0, 1, 3, 2))
        scores = tf.matmul((queries * scale), keys)
        if mask is not None:
            scores = scores + mask
        scores = tf.nn.softmax(scores, axis=-1)
        values_hat = tf.reshape(tf.transpose(tf.matmul(scores, values), (0, 2, 1, 3)), (B, L, -1))

        return self.out_proj(values_hat), (keys, values)


class LlamaEncoderLayer:
    def __init__(self, dims: int, mlp_dims: int, num_heads: int, dtype='float32'):
        self.attention = LlamaAttention(dims, num_heads, dtype)

        self.norm1 = RMSNorm(dims, dtype=dtype)
        self.norm2 = RMSNorm(dims, dtype=dtype)

        self.linear1 = dense(mlp_dims, dims, use_bias=False, dtype=dtype)
        self.linear2 = dense(mlp_dims, dims, use_bias=False, dtype=dtype)
        self.linear3 = dense(dims, mlp_dims, use_bias=False, dtype=dtype)
        
        self.output_size = self.linear3.output_size
        self.param = [self.attention.param, self.norm1.param, self.norm2.param, self.linear1.param,
                      self.linear2.param, self.linear3.param]

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = activation_dict['silu'](a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


class RoPE:
    def __init__(self, dims: int, traditional: bool = False):
        self.dims = dims
        self.traditional = traditional

    def _compute_rope(self, costheta, sintheta, x):
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = tf.concat([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = tf.concat([rx1, rx2], axis=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            raise NotImplementedError(
                "RoPE doesn't implement partial traditional application"
            )

        rx = tf.concat([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = tf.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, device=x.device, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return tf.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        dtype=tf.float32,
    ):
        D = D // 2
        positions = tf.range(offset, N, dtype=dtype)
        freqs = tf.math.exp(
            -tf.range(0, D, dtype=dtype) * (tf.math.log(base) / D)
        )
        theta = tf.reshape(positions, (-1, 1)) * tf.reshape(freqs, (1, -1))
        costheta = tf.math.cos(theta)
        sintheta = tf.math.sin(theta)

        return costheta, sintheta


class RMSNorm:
    def __init__(self, dims: int, epsilon: float = 1e-6, dtype='float32'):
        self.gamma = initializer_((dims,), 'ones', dtype)
        self.epsilon = epsilon
        self.param = [self.gamma]

    def __call__(self, x):
        n = tf.math.rsqrt(tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True) + self.epsilon)
        return self.gamma * x * n