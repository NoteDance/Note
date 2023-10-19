import tensorflow as tf
from Note.nn.layer.dense import dense
from typing import Optional
from Note.nn.Module import Module


class multihead_attention:
    def __init__(self, n_head: int, input_size=None, kv_cache=None, weight_initializer='Xavier', bias_initializer='zeros', use_bias=True, dtype='float32'):
        self.n_head = n_head
        self.input_size=input_size
        self.kv_cache=kv_cache
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.use_bias=use_bias
        self.dtype=dtype
        if input_size!=None:
            self.query = dense(input_size,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.key = dense(input_size,input_size,weight_initializer=weight_initializer,use_bias=use_bias,dtype=dtype)
            self.value = dense(input_size,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.out = dense(input_size,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.param = [self.query.param,self.key.param,self.value.param,self.out.param]
            Module.param.extend(self.param)
    
    
    def build(self):
        self.query = dense(self.input_size,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.key = dense(self.input_size,self.input_size,weight_initializer=self.weight_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.value = dense(self.input_size,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.out = dense(self.input_size,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.param = [self.query.param,self.key.param,self.value.param,self.out.param]
        Module.param.extend(self.param)
        return
    
    
    def qkv_attention(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: Optional[tf.Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = tf.reshape(q, [n_batch, n_ctx, self.n_head, -1])
        q = tf.transpose(q, [0, 2, 1, 3]) * scale
        k = tf.reshape(k, [n_batch, n_ctx, self.n_head, -1])
        k = tf.transpose(k, [0, 2, 3, 1]) * scale
        v = tf.reshape(v, [n_batch, n_ctx, self.n_head, -1])
        v = tf.transpose(v, [0, 2, 1, 3])

        qk = tf.matmul(q, k)
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = tf.nn.softmax(qk)
        return tf.reshape(tf.transpose(tf.matmul(w, v), [0, 2, 1, 3]), [n_batch, n_ctx, n_state]), qk
    
    
    def output(
        self,
        x: tf.Tensor,
        xa: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
    ):
        if x.dtype!=self.dtype:
            x=tf.cast(x,self.dtype)
        if xa is not None and xa.dtype!=self.dtype:
            xa=tf.cast(xa,self.dtype)
        
        if self.input_size==None:
            self.input_size=x.shape[-1]
            self.build()
            
        q = self.query.output(x)

        if self.kv_cache is None or xa is None or self.key not in self.kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key.output(x if xa is None else xa)
            v = self.value.output(x if xa is None else xa)
            if self.kv_cache is not None:
                self.kv_cache[self.key] = k
                self.kv_cache[self.value] = v
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = self.kv_cache[self.key]
            v = self.kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out.output(wv), qk
