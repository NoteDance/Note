import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.RoPE import RoPE
from Note.nn.initializer import initializer_
from dataclasses import dataclass
from Note.nn.Model import Model


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True


class RMSNorm:
    def __init__(self, dims: int, eps: float = 1e-5):
        self.weight = initializer_((dims,), 'ones', 'float32')
        self.eps = eps

    def _norm(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.math.square(x), -1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = tf.cast(self._norm(tf.cast(x, 'float32')), x.dtype)
        return self.weight * output


class Attention:
    def __init__(self, args: ModelArgs):
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = dense(args.n_heads * args.head_dim, args.dim, use_bias=False)
        self.wk = dense(args.n_kv_heads * args.head_dim, args.dim, use_bias=False)
        self.wv = dense(args.n_kv_heads * args.head_dim, args.dim, use_bias=False)
        self.wo = dense(args.dim, args.n_heads * args.head_dim, use_bias=False)
        self.rope = RoPE(
            args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x,
        mask = None,
        cache = None,
    ):
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = tf.transpose(tf.reshape(queries, (B, L, self.n_heads, -1)), (0, 2, 1, 3))
        keys = tf.transpose(tf.reshape(keys, (B, L, self.n_kv_heads, -1)), (0, 2, 1, 3))
        values = tf.transpose(tf.reshape(values, (B, L, self.n_kv_heads, -1)), (0, 2, 1, 3))

        def repeat(a):
            a = tf.concat([tf.expand_dims(a, 2)] * self.repeats, axis=2)
            return tf.reshape(a, [B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = tf.concat([key_cache, keys], axis=2)
            values = tf.concat([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = tf.matmul((queries * self.scale), tf.transpose(keys, (0, 1, 3, 2)))
        if mask is not None:
            scores += mask
        scores = tf.cast(tf.nn.softmax(tf.cast(scores, 'float32'), axis=-1), scores.dtype)
        output = tf.reshape(tf.transpose((tf.matmul(scores, values), (0, 2, 1, 3)), (B, L, -1)))
        return self.wo(output), (keys, values)


class FeedForward:
    def __init__(self, args: ModelArgs):
        self.w1 = dense(args.hidden_dim, args.dim, use_bias=False)
        self.w2 = dense(args.dim, args.hidden_dim, use_bias=False)
        self.w3 = dense(args.hidden_dim, args.dim, use_bias=False)

    def __call__(self, x):
        return self.w2(tf.nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock:
    def __init__(self, args: ModelArgs):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x,
        mask = None,
        cache = None,
    ):
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Llama(Model):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = initializer_((args.vocab_size, args.dim), 'normal', 'float32')
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = dense(args.vocab_size, args.dim, use_bias=False)

    def __call__(self, x):
        mask = tf.linalg.band_part(tf.ones((x.shape[0], x.shape[1], x.shape[1])), -1, 0)
        mask = tf.cast(mask, self.tok_embeddings.dtype)

        x = self.tok_embeddings(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, x, temp=1.0):
        def sample(logits):
            if temp == 0:
                return tf.argmax(logits, axis=-1)
            else:
                return tf.random.categorical(logits * (1 / temp))

        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = tf.linalg.band_part(tf.ones((x.shape[0], x.shape[1], x.shape[1])), -1, 0)
        mask = tf.cast(mask, self.tok_embeddings.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.tok_embeddings(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            # We store the per layer cache in a simple python list
            cache.append(c)
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.output(x[:, -1])
        y = sample(y)

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.tok_embeddings(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = sample(self.output(x[:, -1]))

            yield y