import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.conv1d import conv1d
from Note.nn.layer.zeropadding1d import zeropadding1d
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.initializer import initializer_
from Note.nn.Layers import Layers
import base64
import gzip
import numpy as np


class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = tf.math.exp(-log_timescale_increment * np.arange(channels // 2))
    scaled_time = np.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return tf.concat([tf.math.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)


class cached_attention:
  def __init__(self,n_state,n_head):
      self.n_state=n_state
      self.n_head=n_head
      self.query_dense=dense(n_state,n_state)
      self.key_dense=dense(n_state,n_state,use_bias=False)
      self.value_dense=dense(n_state,n_state)
      self.output_dense=dense(n_state,n_state)


  def _update_cache(self, key, value, cache):
    if key.shape[1] > self.dims.n_text_ctx:
        cache["key"] = key
    else:
        key = tf.concat([tf.cast(cache["key"], key.dtype), key], axis=1)
        cache["key"] = key
    if value.shape[1] > self.dims.n_text_ctx:
        cache["value"] = value
    else:
        value = tf.concat([tf.cast(cache["value"], value.dtype), value], axis=1)
        cache["value"] = value

    return key, value


  def __call__(self,
           x,
           xa,
           mask=None,
           cache=None,
           ):
    query = self.query_dense(x)
    n_batch, n_ctx, n_state = query.shape
    scale = (n_state // self.n_head) ** -0.25
    query = tf.reshape(query, [n_batch, n_ctx, self.n_head, -1])

    key = self.key_dense(x if xa is None else xa)
    n_batch, n_ctx, n_state = key.shape
    key = tf.reshape(key, [n_batch, n_ctx, self.n_head, -1])

    value = self.value_dense(x if xa is None else xa)
    n_batch, n_ctx, n_state = key.shape
    value = tf.reshape(value, [n_batch, n_ctx, self.n_head, -1])

    if cache:
      key, value = self._update_cache(key, value, cache)
     
    query = tf.transpose(query, [0, 2, 1, 3]) * scale
    key = tf.transpose(key, [0, 2, 3, 1]) * scale
    value = tf.transpose(value, [0, 2, 1, 3])

    qk = tf.matmul(query, key)
    if mask is not None:
        qk = qk + mask[:n_ctx, :n_ctx]
    w = tf.nn.softmax(qk)
    wv = tf.reshape(tf.transpose(tf.matmul(w, value), [0, 2, 1, 3]), [n_batch, n_ctx, n_state])
    return self.output_dense(wv), qk


class ResidualAttentionBlock:
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        self.attn = cached_attention(n_state, n_head)
        self.attn_ln = layer_norm(n_state)

        self.cross_attn = (
            cached_attention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = layer_norm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = Layers()
        self.mlp.add(dense(n_mlp, n_state))
        self.mlp.add(tf.nn.gelu)
        self.mlp.add(dense(n_state, n_mlp))
        
        self.mlp_ln = layer_norm(n_state)

    def __call__(
        self,
        x,
        xa = None,
        mask = None,
        kv_cache = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder:
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        self.conv1 = conv1d(filters=n_state, input_size=n_mels, kernel_size=3)
        self.zeropadding1d1 = zeropadding1d(padding=1)
        self.conv2 = conv1d(filters=n_state, input_size=n_state, kernel_size=3, stride=2)
        self.zeropadding1d2 = zeropadding1d(padding=1)
        self.positional_embedding=sinusoids(n_ctx, n_state)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = layer_norm(n_state)


    def __call__(self, x):
        x = self.conv1(x)
        x = self.zeropadding1d1(x)
        x = tf.nn.gelu(x)
        x = self.conv2(x)
        x = self.zeropadding1d2(x)
        x = tf.nn.gelu(x)
        x = tf.transpose(x, [0, 2, 1])

        x = x + tf.cast(self.positional_embedding, x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder:
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        self.token_embedding = initializer_([n_vocab, n_state], 'zeros', 'float32')
        self.positional_embedding = initializer_([n_ctx, n_state], 'zeros', 'float32')

        self.blocks = [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)]
        
        self.ln = layer_norm(n_state)

        self.mask = tf.math.multiply(tf.linalg.band_part(tf.ones((n_ctx, n_ctx)), -1, 0), tf.fill((n_ctx, n_ctx), tf.float32.min))


    def __call__(self, x, xa, kv_cache = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            tf.matmul(x, self.token_embedding)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = tf.matmul(x, tf.cast(tf.transpose(self.token_embedding, perm=[0, 1])), x.dtype)

        return logits


class Whisper:
    def __init__(self, dims: ModelDimensions):
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = tf.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=tf.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = tf.sparse.from_dense(all_heads)


    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = tf.reshape(tf.convert_to_tensor(array), (self.dims.n_text_layer, self.dims.n_text_head))
        self.alignment_heads = tf.sparse.from_dense(mask)
        return


    def embed_audio(self, mel):
        return self.encoder(mel)


    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)


    def __call__(self, mel, tokens, cache):
        return self.decoder(tokens, self.encoder(mel), cache)