import tensorflow as tf
import numpy as np
from Note.nn.initializer import initializer_
from Note.nn.Module import Module

class hparams:
    n_vocab=0
    n_ctx=1024
    n_embd=768
    n_head=12
    n_layer=12

class GPT2:
    def __init__(self,one_hot=True):
        self.one_hot=one_hot
        self.norm=norm()
        self.block={}
        self.opt=tf.keras.optimizers.Adam()
        self.param=Module.param
        self.flag=0
        
    def fp(self, X, past=None):
        results = {}
        batch, sequence = shape_list(X)
        
        if self.flag==0:
            self.wpe = initializer_([hparams.n_ctx, hparams.n_embd], ['normal',0.0,0.01], 'float32')
            self.wte = initializer_([hparams.n_vocab, hparams.n_embd], ['normal',0.0,0.02], 'float32')
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(self.wte, X) + tf.gather(self.wpe, self.positions_for(X, past_length))
    
        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            if self.flag==0:
                self.block[layer]=block()
            h, present = self.block[layer].output(h, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = self.norm.output(h)
    
        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, self.wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        self.flag=1
        return results
    
    def loss(self, output, labels):
        # Get the logits from the model output
        logits = output['logits']
        
        # Convert the logits to probabilities
        probs = softmax(logits, axis=-1)
        
        if self.one_hot:
            # Convert the labels to one-hot vectors
            labels = tf.one_hot(labels, depth=hparams.n_vocab)
        
        # Compute the cross entropy loss
        loss = -tf.reduce_sum(labels * tf.math.log(probs), axis=-1)
        loss = tf.reduce_mean(loss)
        
        return loss

    def past_shape(self, hparams, batch_size=None, sequence=None):
        return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

    def expand_tile(self, value, size):
        """Add a new axis of given size."""
        value = tf.convert_to_tensor(value)
        ndims = value.shape.ndims
        return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

    def positions_for(self, tokens, past_length):
        batch_size = tf.shape(tokens)[0]
        nsteps = tf.shape(tokens)[1]
        return self.expand_tile(past_length + tf.range(nsteps), batch_size)

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def attention_mask(nd, ns, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

class conv1d:
    def __init__(self):
        self.flag=0
    
    def output(self, x, nf, w_init_stdev=0.02):
        *start, nx = shape_list(x)
        if self.flag==0:
            self.w = initializer_([1, nx, nf], ['normal',0.0,w_init_stdev], 'float32')
            self.b = initializer_([nf],'zeros','float32')
            self.flag=1
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(self.w, [-1, nf]))+self.b, start+[nf])
        return c

class norm:
    def __init__(self):
        self.flag=0
        
    def output(self, x, axis=-1, epsilon=1e-5):
        """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
        n_state = x.shape[-1]
        if self.flag==0:
            self.g = initializer_([n_state],'ones','float32')
            self.b = initializer_([n_state],'zeros','float32')
            self.flag=1
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + epsilon)
        x = x*self.g + self.b
        return x

class attn:
    def __init__(self):
        self.flag=0
    
    def output(self, x, n_state, past, hparams):
        if self.flag==0:
            self.conv1d1=conv1d()
            self.conv1d2=conv1d()
            self.flag=1
        assert x.shape.ndims == 3  # Should be [batch, sequence, features]
        assert n_state % hparams.n_head == 0
        if past is not None:
            assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

        def split_heads(x):
            # From [batch, sequence, features] to [batch, heads, sequence, features]
            return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

        def merge_heads(x):
            # Reverse of split_heads
            return merge_states(tf.transpose(x, [0, 2, 1, 3]))

        def mask_attn_weights(w):
            # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
            _, _, nd, ns = shape_list(w)
            b = attention_mask(nd, ns, dtype=w.dtype)
            b = tf.reshape(b, [1, 1, nd, ns])
            w = w*b - tf.cast(1e10, w.dtype)*(1-b)
            return w

        def multihead_attn(q, k, v):
            # q, k, v have shape [batch, heads, sequence, features]
            w = tf.matmul(q, k, transpose_b=True)
            w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))

            w = mask_attn_weights(w)
            w = softmax(w)
            a = tf.matmul(w, v)
            return a

        c = self.conv1d1.output(x, n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = self.conv1d2.output(a, n_state)
        return a, present
    
class mlp:
    def __init__(self):
        self.flag=0
    
    def output(self, x, n_state, hparams):
        if self.flag==0:
            self.conv1d1=conv1d()
            self.conv1d2=conv1d()
            self.flag=1
        nx = x.shape[-1]
        h = gelu(self.conv1d1.output(x, n_state))
        h2 = self.conv1d2.output(h, nx)
        return h2

class block:
    def __init__(self):
        self.flag=0
        
    def output(self, x, past, hparams):
        if self.flag==0:
            self.norm1=norm()
            self.norm2=norm()
            self.attn=attn()
            self.mlp=mlp()
            self.flag=1
        nx = x.shape[-1]
        a, present = self.attn.output(self.norm1.output(x), nx, past=past, hparams=hparams)
        x = x + a
        m = self.mlp.output(self.norm2.output(x), nx*4, hparams=hparams)
        x = x + m
        return x, present