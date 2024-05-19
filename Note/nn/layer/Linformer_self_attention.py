import tensorflow as tf
from Note.nn.layer.dense import dense
import math
from Note.nn.Model import Model


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor = tf.random.uniform(shape=tensor.shape, minval=-std, maxval=std)
    return tensor


class Linformer_self_attention:
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0., dtype='float32'):
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head
        
        self.dtype=dtype
        
        self.param=[]

        self.to_q = dense(dim_head * heads, dim, use_bias = False, dtype=dtype)
        self.param.append(self.to_q.param)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = dense(kv_dim, dim, use_bias = False, dtype=dtype)
        self.proj_k = tf.Variable(init_(tf.zeros([seq_len, k], dtype=dtype)))
        self.param.append(self.to_k.param)
        self.param.append(self.proj_k)

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = dense(kv_dim, dim, use_bias = False, dtype=dtype)
            self.proj_v = tf.Variable(init_(tf.zeros([seq_len, k], dtype=dtype)))
            self.param.append(self.to_v.param)
            self.param.append(self.proj_v)

        self.dropout = tf.nn.dropout
        self.to_out = dense(dim, dim_head * heads, dtype=dtype)
        self.param.append(self.to_out.param)
        self.dropout_rate=dropout
        Model.param.extend(self.param)

    def __call__(self, x, context = None, train_flag=True):
        if x.dtype!=self.dtype:
            x=tf.cast(x,self.dtype)
            
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
        
        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'
        
        queries = self.to_q(x)
        
        proj_seq_len = lambda args: tf.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = tf.reshape(queries, shape=(b, n, h, -1))
        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        
        merge_key_values = lambda t: tf.transpose(tf.reshape(t, [b, k, -1, d_h]), [0, 2, 1, 3])
        keys, values = map(merge_key_values, (keys, values))
        
        # attention
        
        dots = tf.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = tf.nn.softmax(dots,axis=-1)
        if train_flag:
            attn = self.dropout(attn,self.dropout_rate)
        out = tf.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = tf.reshape(tf.transpose(out, perm=[0, 2, 1, 3]), shape=(b, n, -1))
        return self.to_out(out)