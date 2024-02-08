import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.masked_softmax import masked_softmax


class multichannel_attention:
    """Multi-channel Attention layer.
    
    Introduced in, [Generating Representative Headlines for News Stories
    ](https://arxiv.org/abs/2001.09386). Expects multiple cross-attention
    target sequences.
    
    Call args:
      query: Query `Tensor` of shape `[B, T, dim]`.
      value: Value `Tensor` of shape `[B, A, S, dim]`, where A denotes the
      context_attention_weights: Context weights of shape `[B, N, T, A]`, where N
        is the number of attention heads. Combines multi-channel sources
        context tensors according to the distribution among channels.
      key: Optional key `Tensor` of shape `[B, A, S, dim]`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: A boolean mask of shape `[B, T, S]`, that prevents attention
        to certain positions.
    """
    
    def __init__(self, n_head, key_dim, value_dim=None, input_size=None, dropout_rate=0.0, weight_initializer='Xavier', bias_initializer='zeros', use_bias=True, dtype='float32'):
      self.n_head=n_head
      self.key_dim=key_dim
      self.value_dim=value_dim if value_dim else key_dim
      self.input_size=input_size
      self.dropout_rate=dropout_rate
      self.weight_initializer=weight_initializer
      self.bias_initializer=bias_initializer
      self.use_bias=use_bias
      self.dtype=dtype
      self._masked_softmax=masked_softmax(mask_expansion_axes=[2])
      if input_size is not None:
          self.query_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.key_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.value_dense=dense(n_head*value_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.output_dense=dense(input_size,n_head*value_dim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param]
    
    
    def build(self):
        self.query_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.key_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.value_dense=dense(self.n_head*self.value_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.output_dense=dense(self.input_size,self.n_head*self.value_dim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param]
        return

    
    def __call__(self,
             query,
             value,
             key=None,
             context_attention_weights=None,
             attention_mask=None,
             train_flag=True):
      if key is None:
        key = value
      
      if query.dtype!=self.dtype:
          query=tf.cast(query,self.dtype)
      if value.dtype!=self.dtype:
          value=tf.cast(value,self.dtype)
      if key.dtype!=self.dtype:
          key=tf.cast(key,self.dtype) 
    
      if self.input_size==None:
          self.input_size=query.shape[-1]
          self.build()
    
      # Scalar dimensions referenced here:
      #   B = batch size (number of stories)
      #   A = num_docs (number of docs)
      #   F = target sequence length
      #   T = source sequence length
      #   N = `num_attention_heads`
      #   H = `size_per_head`
      # `query_tensor` = [B, F, N ,H]
      query_tensor = self.query_dense(query)
      n_batch, n_ctx, n_state = query_tensor.shape
      query_tensor = tf.reshape(query_tensor, [n_batch, n_ctx, self.n_head, -1])
    
      # `key_tensor` = [B, A, T, N, H]
      key_tensor = self.key_dense(key)
      n_batch, n_doc, n_ctx, n_state = key_tensor.shape
      key_tensor = tf.reshape(key_tensor, [n_batch, n_doc, n_ctx, self.n_head, -1])
    
      # `value_tensor` = [B, A, T, N, H]
      value_tensor = self.value_dense(value)
      n_batch, n_doc, n_ctx, n_state = value_tensor.shape
      value_tensor = tf.reshape(value_tensor, [n_batch, n_doc, n_ctx, self.n_head, -1])
    
      # Take the dot product between "query" and "key" to get the raw
      # attention scores.
      attention_scores = tf.einsum("BATNH,BFNH->BANFT", key_tensor, query_tensor)
      attention_scores = tf.multiply(attention_scores,
                                     1.0 / tf.math.sqrt(float(self.input_size)))
    
      # Normalize the attention scores to probabilities.
      # `attention_probs` = [B, A, N, F, T]
      attention_probs = self._masked_softmax(attention_scores, attention_mask)
    
      # This is actually dropping out entire tokens to attend to, which might
      # seem a bit unusual, but is taken from the original Transformer paper.
      if train_flag:
          attention_probs = tf.nn.dropout(attention_probs,self.dropout_rate)
    
      # `context_layer` = [B, F, N, H]
      context_layer = tf.einsum("BANFT,BATNH->BAFNH", attention_probs,
                                value_tensor)
      attention_output = tf.einsum("BNFA,BAFNH->BFNH", context_attention_weights,
                                   context_layer)
      B, F, _, _ = attention_output.shape
      attention_output = tf.reshape(attention_output, [B, F, -1])
      attention_output = self.output_dense(attention_output)
      return attention_output