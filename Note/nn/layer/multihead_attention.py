import tensorflow as tf
from Note import nn


class multihead_attention:
    def __init__(self, n_head: int, input_size=None, kdim=None, vdim=None, dropout=0.0, weight_initializer='Xavier', bias_initializer='zeros', use_bias=True, dtype='float32'):
        self.n_head = n_head
        self.input_size = input_size
        self.kdim = kdim if kdim is not None else input_size
        self.vdim = vdim if vdim is not None else input_size
        self.dropout_rate = dropout
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.use_bias=use_bias
        self.dtype=dtype
        self.dropout=nn.dropout(dropout)
        if input_size is not None:
            self.query = nn.dense(input_size,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.key = nn.dense(input_size,self.kdim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.value = nn.dense(input_size,self.vdim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.out = nn.dense(input_size,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.param = [self.query.param,self.key.param,self.value.param,self.out.param]
    
    
    def build(self):
        self.query = nn.dense(self.input_size,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.key = nn.dense(self.input_size,self.kdim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.value = nn.dense(self.input_size,self.vdim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.out = nn.dense(self.input_size,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.param = [self.query.param,self.key.param,self.value.param,self.out.param]
        return
    
    
    def qkv_attention(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask = None
    ):
        n_batch_q, n_ctx_q, n_state = q.shape
        n_batch_k, n_ctx_k, _ = k.shape
        n_batch_v, n_ctx_v, _ = v.shape
        q = tf.reshape(q, [n_batch_q, n_ctx_q, self.n_head, -1])
        q = tf.transpose(q, [0, 2, 1, 3])
        q=tf.multiply(q, 1.0 / tf.math.sqrt(float(self.kdim)))
        k = tf.reshape(k, [n_batch_k, n_ctx_k, self.n_head, -1])
        k = tf.transpose(k, [0, 2, 3, 1])
        v = tf.reshape(v, [n_batch_v, n_ctx_v, self.n_head, -1])
        v = tf.transpose(v, [0, 2, 1, 3])

        qk = tf.matmul(q, k)
        if mask is not None:
            qk = qk + mask[:n_ctx_q, :n_ctx_q]

        w = tf.nn.softmax(qk)
        if self.dropout_rate:
            w = self.dropout(w)
        return tf.reshape(tf.transpose(tf.matmul(w, v), [0, 2, 1, 3]), [n_batch_q, n_ctx_q, n_state]), qk
    
    
    def __call__(
        self,
        target,
        source = None,
        mask = None,
    ):
        if target.dtype!=self.dtype:
            target=tf.cast(target,self.dtype)
        if source is not None and source.dtype!=self.dtype:
            source=tf.cast(source,self.dtype)
        
        if self.input_size==None:
            self.input_size=target.shape[-1]
            self.build()
            
        q = self.query(target)
        k = self.key(target if source is None else source)
        v = self.value(target if source is None else source)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk
