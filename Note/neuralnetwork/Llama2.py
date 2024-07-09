import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.dropout import dropout
from Note.nn.initializer import initializer
from Note.nn.parallel.optimizer import AdamW
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Model import Model
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    lr: float = 0.0003
    weight_decay: float = 0.1
    device: str = 'GPU'


class RMSNorm:
    def __init__(self, dim: int, eps: float):
        self.eps = eps
        self.weight = initializer((dim,), 'ones', 'float32')

    def _norm(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.math.pow(x, 2), -1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = tf.cast(self._norm(tf.cast(x, 'float32')), x.dtype)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (tf.cast(tf.range(0, dim, 2)[: (dim // 2)], 'float32') / dim))
    t = tf.range(end)  # type: ignore
    freqs = tf.cast(tf.experimental.numpy.outer(t, freqs), 'float32')  # type: ignore
    freqs_cos = tf.math.cos(freqs)  # real part
    freqs_sin = tf.math.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return tf.reshape(freqs_cis, shape)

def apply_rotary_emb(
    xq,
    xk,
    freqs_cos,
    freqs_sin
):

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = tf.unstack(tf.reshape(tf.cast(xq, 'float32'), (xq.shape[-1] // 2, 2)), axis=-1)
    xk_r, xk_i = tf.unstack(tf.reshape(tf.cast(xk,  'float32'), (xk.shape[-1] // 2, 2)), axis=-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = tf.stack([xq_out_r, xq_out_i], axis=-1)
    shape = xq_out.shape
    xq_out = tf.reshape(xq_out, [-1, shape[1], shape[2], shape[3] * shape[4]])
    xk_out = tf.stack([xk_out_r, xk_out_i], axis=-1)
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
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = dense(args.n_heads * self.head_dim, args.dim, weight_initializer=['normal', 0.0, 0.02], use_bias=False)
        self.wk = dense(self.n_kv_heads * self.head_dim, args.dim, weight_initializer=['normal', 0.0, 0.02], use_bias=False)
        self.wv = dense(self.n_kv_heads * self.head_dim, args.dim, weight_initializer=['normal', 0.0, 0.02], use_bias=False)
        self.wo = dense(args.dim, args.n_heads * self.head_dim, weight_initializer=['normal', 0.0, 0.02/math.sqrt(2 * args.n_layers)], use_bias=False)
        self.attn_dropout = dropout(args.dropout)
        self.resid_dropout = dropout(args.dropout)
        self.mask = tf.fill((args.max_seq_len, args.max_seq_len), float("-inf"))
        self.mask = tf.linalg.band_part(self.mask, 0, -1)
        self.mask = tf.linalg.set_diag(self.mask, tf.zeros(args.max_seq_len))
        self.mask = tf.reshape(self.mask, (1, 1, *self.mask.shape))

    def __call__(
        self,
        x,
        freqs_cos,
        freqs_sin,
        train_flag=True
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = tf.reshape(xq, (bsz, seqlen, self.n_local_heads, self.head_dim))
        xk = tf.reshape(xk, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        xv = tf.reshape(xv, (bsz, seqlen, self.n_local_kv_heads, self.head_dim))

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = tf.transpose(xq, (0, 2, 1, 3))  # (bs, n_local_heads, seqlen, head_dim)
        xk = tf.transpose(xk, (0, 2, 1, 3))
        xv = tf.transpose(xv, (0, 2, 1, 3))

        scores = tf.matmul(xq, tf.transpose(xk, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        assert hasattr(self, 'mask')
        scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = tf.cast(tf.nn.softmax(tf.cast(scores, 'float32'), axis=-1), xq.dtype)
        scores = self.attn_dropout(scores, train_flag)
        output = tf.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = tf.reshape(tf.transpose(output, (0, 2, 1, 3)), (bsz, seqlen, -1))

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output, train_flag)
        return output


class FeedForward:
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, drop_rate: float):
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = dense(hidden_dim, dim, weight_initializer=['normal', 0.0, 0.02], use_bias=False)
        self.w2 = dense(dim, hidden_dim, weight_initializer=['normal', 0.0, 0.02], use_bias=False)
        self.w3 = dense(hidden_dim, dim, weight_initializer=['normal', 0.0, 0.02/math.sqrt(2 * ModelArgs.n_layers)], use_bias=False)
        self.dropout = dropout(drop_rate)

    def __call__(self, x, train_flag=True):
        return self.dropout(self.w2(tf.nn.silu(self.w1(x)) * self.w3(x)), train_flag)


class TransformerBlock:
    def __init__(self, layer_id: int, args: ModelArgs):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            drop_rate=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x, freqs_cos, freqs_sin, train_flag=True):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, train_flag)
        out = h + self.feed_forward(self.ffn_norm(h), train_flag)
        return out


class Llama2(Model):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.dropout = dropout(params.dropout)
        self.layers = []
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.head = self.dense(params.vocab_size, params.dim, weight_initializer=['normal', 0.0, 0.02], use_bias=False)

        # some useful precompute for the RoPE relative positional embeddings
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        
        self.optimizer=AdamW(lr=self.params.lr, beta1=0.9, beta2=0.95)
        self.device=self.params.device
        self.km=0

    def fp(self, tokens, p = None):
        self.apply_decay('dense_weight', ModelArgs.weight_decay, False)
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                _bsz, seqlen = tokens.shape
                h = tf.gather(tf.transpose(self.output.weight), tokens)
                h = self.dropout(h)
                freqs_cos = self.freqs_cos[:seqlen]
                freqs_sin = self.freqs_sin[:seqlen]
        
                for layer in self.layers:
                    h = layer(h, freqs_cos, freqs_sin)
                h = self.norm(h)
        
                logits = self.output(h)

                return logits
        else:
            _bsz, seqlen = tokens.shape
            h = tf.gather(self.tok_embeddings, tokens)
            h = self.dropout(h, self.km)
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]
    
            for layer in self.layers:
                h = layer(h, freqs_cos, freqs_sin, self.km)
            h = self.norm(h)
        
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
    
            return logits
    
    def loss(self, output, labels, p):
        with tf.device(assign_device(p,self.device)):
            output = tf.reshape(output, [-1, output.shape[-1]])
            labels = tf.reshape(labels, [-1])
            mask = tf.not_equal(labels, -1)
            loss = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output), mask)
            return tf.reduce_mean(loss)
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
            return tape,output,loss
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,self.device)):
            self.apply_decay('dense_weight', ModelArgs.weight_decay)
            param=self.optimizer(gradient,self.param,self.bc[0])
            return param

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                idx_next = tf.math.argmax(logits, axis=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    k = tf.minimum(top_k, logits.shape[-1])
                    v, _ = tf.math.top_k(logits, k=k, sorted=True)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = tf.nn.softmax(logits, dim=-1)
                idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1)
            # append sampled index to the running sequence and continue
            idx = tf.concat((idx, idx_next), axis=1)

        return idx