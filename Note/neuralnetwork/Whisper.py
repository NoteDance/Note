import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.conv1d import conv1d
from Note.nn.layer.zeropadding1d import zeropadding1d
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.initializer import initializer_
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Module import Module
import base64
import gzip
import numpy as np
from typing import Union


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


class LayerNorm:
    def __init__(self, n_state):
        self.layer_norm = layer_norm(n_state)
        
    def __call__(self, x):
        return tf.cast(self.layer_norm(tf.cast(x, 'float32')), x.dtype)


class MultiHeadAttention:
    def __init__(self, n_state: int, n_head: int):
        self.n_head = n_head
        self.query = dense(n_state, n_state)
        self.key = dense(n_state, n_state, use_bias=False)
        self.value = dense(n_state, n_state)
        self.out = dense(n_state, n_state)

    def __call__(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        q = self.query(x)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                k = tf.concat([kv_cache[0], k], axis=1)
                v = tf.concat([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            k = self.key(xa)
            v = self.value(xa)
        else:
            k, v = kv_cache

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), (k, v), qk

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        qk = tf.matmul(q, k)
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = tf.cast(qk, tf.float32)

        w = tf.cast(tf.nn.softmax(qk, axis=-1), q.dtype)
        out = tf.transpose(tf.matmul(w, v), (0, 2, 1, 3))
        out = tf.reshape(out, (n_batch, n_ctx, n_state))
        return out, qk


class ResidualAttentionBlock:
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp1 = dense(n_mlp, n_state)
        self.mlp2 = dense(n_state, n_mlp)
        self.mlp_ln = LayerNorm(n_state)

    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
        x += y
        cross_qk = None
        if self.cross_attn:
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=cross_kv
            )
            x += y
        x = x + tf.cast(self.mlp2(tf.nn.gelu(self.mlp1(self.mlp_ln(x))), x.dtype))
        return x, (kv, cross_kv), cross_qk


class AudioEncoder:
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype = tf.float16,
    ):
        self.conv1 = conv1d(filters=n_state, input_size=n_mels, kernel_size=3)
        self.zeropadding1d1 = zeropadding1d(padding=1)
        self.conv2 = conv1d(filters=n_state, input_size=n_state, kernel_size=3, stride=2)
        self.zeropadding1d2 = zeropadding1d(padding=1)
        self._positional_embedding = tf.cast(sinusoids(n_ctx, n_state), dtype)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = LayerNorm(n_state)

    def __call__(self, x):
        x = tf.cast(tf.nn.gelu(self.conv1(x)), x.dtype)
        x = tf.cast(tf.nn.gelu(self.conv2(x)), x.dtype)
        assert x.shape[1:] == self._positional_embedding.shape, "incorrect audio shape"
        x = x + self._positional_embedding

        for block in self.blocks:
            x, _, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder:
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype = tf.float16,
    ):
        self.token_embedding = initializer_([n_vocab, n_state], 'normal', 'float32')
        self.positional_embedding = initializer_([n_ctx, n_state], 'zeros', 'float32')

        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        self.ln = LayerNorm(n_state)
        self._mask = tf.fill((3, 3), float("-inf"))
        self._mask = tf.linalg.band_part(self.mask, 0, -1)
        self._mask = tf.linalg.set_diag(self.mask, tf.zeros(3))
        self._mask = tf.cast(self._mask, dtype)

    def __call__(self, x, xa, kv_cache=None):
        """
        x : shape = (batch_size, <= n_ctx)
            the text tokens
        xa : shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x = (
            tf.gather(self.token_embedding, x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        cross_qk = [None] * len(self.blocks)
        for e, block in enumerate(self.blocks):
            x, kv_cache[e], cross_qk[e] = block(
                x, xa, mask=self._mask, kv_cache=kv_cache[e]
            )

        x = self.ln(x)
        return tf.matmul(x, tf.transpose(self.token_embedding)), kv_cache, cross_qk


class Whisper:
    def __init__(self, dims: ModelDimensions, dtype = tf.float16, device = 'GPU'):
        Module.init()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), dtype=bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = tf.transpose(tf.cast(tf.where(all_heads != 0), dtype=tf.int32))
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer=Adam()
        self.param = Module.param
        self.device=device

    def set_alignment_heads(self, dump: Union[bytes, np.ndarray]):
        if isinstance(dump, np.ndarray):
            self.alignment_heads = tf.convert_to_tensor(dump)
        elif isinstance(dump, bytes):
            array = np.frombuffer(
                gzip.decompress(base64.b85decode(dump)), dtype=bool
            ).copy()
            mask = array.reshape(self.dims.n_text_layer, self.dims.n_text_head)
            self.alignment_heads = tf.transpose(tf.cast(tf.where(mask != 0), dtype=tf.int32))
        else:
            raise ValueError(
                f"Invalid type for `dump`: {type(dump)}. Expected a np.ndarray or base85-encoded bytes containing"
                " alignment_head information"
            )

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)[0]

    def forward_with_cross_qk(self, mel, tokens):
        logits, _, cross_qk = self.decoder(tokens, self.encoder(mel))
        return logits, cross_qk

    def fp(self, data, p=None):
        with tf.device(assign_device(p,self.device)):
            mel = data[0]
            tokens = data[1]
            return self.decoder(tokens, self.encoder(mel))[0]
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,self.device)):
            output = tf.nn.softmax(output)
            loss=self.loss_object(labels,output)
            return loss
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
            return tape,output,loss
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,self.device)):
            param=self.optimizer(gradient,self.param,self.bc[0])
            return param

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)
