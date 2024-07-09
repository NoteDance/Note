import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.initializer import initializer
import collections


class reuse_multihead_attention:
    """MultiHeadAttention layer.
    
    This is an implementation of multi-headed attention as described in the paper
    "Attention is all you Need" (Vaswani et al., 2017).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.
    
    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.
    
    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.
    
    Finally, the result tensor with the last dimension as value_dim can take an
    linear projection and return.
    
    Args:
      num_heads: Number of attention heads.
      key_dim: Size of each attention head for query and key.
      value_dim: Size of each attention head for value.
      dropout: Dropout probability.
      reuse_attention: An integer specifying number of heads to reuse.
        -1 for all heads.
      use_relative_pe: Whether to use relative position bias.
      max_sequence_length: Used to set the size of the relative positin encodings.
      use_bias: Boolean, whether the dense layers use bias vectors/matrices.
      output_shape: The expected shape of an output tensor, besides the batch and
        sequence dims. If not specified, projects back to the key feature dim.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
    
    Call arguments:
      query: Query `Tensor` of shape `(B, T, dim)`.
      value: Value `Tensor` of shape `(B, S, dim)`.
      key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.
      return_attention_scores: A boolean to indicate whether the output should
        be attention output if True, or (attention_output, attention_scores) if
        False. Defaults to False.
      train_flag: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
        Defaults to either using the training mode of the parent layer/model,
        or False (inference) if there is no parent layer.
    
    Returns:
      attention_output: The result of the computation, of shape `(B, T, E)`,
        where `T` is for target sequence shapes and `E` is the query input last
        dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
        are project to the shape specified by `output_shape`.
      attention_scores: [Optional] multi-head attention coeffients over
        attention axes.
    """
    
    def __init__(self,
                 n_head,
                 key_dim,
                 value_dim=None,
                 input_size=None,
                 dropout=0.0,
                 reuse_attention=0,
                 use_relative_pe=False,
                 pe_max_seq_length=512,
                 use_bias=True,
                 attention_axes=None,
                 weight_initializer="Xavier",
                 bias_initializer="zeros",
                 dtype='float32'
                 ):
      self._num_heads = n_head
      self._key_dim = key_dim
      self._value_dim = value_dim if value_dim else key_dim
      self.input_size=input_size
      self._dropout = dropout
      if reuse_attention > self._num_heads or reuse_attention < -1:
        raise ValueError("reuse_attention should be between -1 "
                         "and %d in call to %s." % (self.__class__,
                                                    self._num_heads))
      if reuse_attention == -1:
        reuse_attention = self._num_heads
      self._reuse_heads = reuse_attention
      self._use_relative_pe = use_relative_pe
      self._pe_max_seq_length = pe_max_seq_length
      self._use_bias = use_bias
      if attention_axes is not None and not isinstance(attention_axes,
                                                       collections.abc.Sized):
        self._attention_axes = (attention_axes,)
      else:
        self._attention_axes = attention_axes
      if self._attention_axes is None:
          self._attention_axes = tuple(range(1, 3 - 2))
      else:
          self._attention_axes = tuple(self._attention_axes)
      self.weight_initializer=weight_initializer
      self.bias_initializer=bias_initializer
      self.dtype=dtype
      if input_size!=None:
          self.query_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.key_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.param=[self.query_dense.param,self.key_dense.param]
          self.value_dense = []
          if self._reuse_heads > 0:
              self.value_dense.append(dense(self._reuse_heads*value_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype))
              self.param.append(self.value_dense[0].param)
          if self._reuse_heads < self._num_heads:
              self.value_dense.append(dense((self.num_heads-self._reuse_heads)*value_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype))
              self.param.append(self.value_dense[1].param)
          self.output_dense = []
          if self._reuse_heads > 0:
              self.output_dense.append(dense(input_size,n_head*value_dim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype))
              self.param.append(self.output_dense[0].param)
          if self._reuse_heads < self._num_heads:
              self.output_dense.append(dense(input_size,n_head*value_dim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=self._reuse_heads==0,dtype=dtype))
              self.param.append(self.output_dense[1].param)
          # Use relative PE only if reuse_heads < num_heads.
          if self._use_relative_pe and self._reuse_heads < self._num_heads:
              self._position_embeddings = initializer([
                         1, self._num_heads - self._reuse_heads, 2 * self.
                         _pe_max_seq_length - 1],['truncated_normal',0.2],dtype)
              self.param.append(self._position_embeddings)
    
    def build(self):
        self.query_dense=dense(self._num_heads*self._key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.key_dense=dense(self._num_heads*self._key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.param=[self.query_dense.param,self.key_dense.param]
        self.value_dense = []
        if self._reuse_heads > 0:
            self.value_dense.append(dense(self._reuse_heads*self._value_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype))
            self.param.append(self.value_dense[0].param)
        if self._reuse_heads < self._num_heads:
            self.value_dense.append(dense((self.num_heads-self._reuse_heads)*self._value_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype))
            self.param.append(self.value_dense[1].param)
        self.output_dense = []
        if self._reuse_heads > 0:
            self.output_dense.append(dense(self.input_size,self._num_heads*self._value_dim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype))
            self.param.append(self.output_dense[0].param)
        if self._reuse_heads < self._num_heads:
            self.output_dense.append(dense(self.input_size,self._num_heads*self._value_dim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self._reuse_heads==0,dtype=self.dtype))
            self.param.append(self.output_dense[1].param)
        # Use relative PE only if reuse_heads < num_heads.
        if self._use_relative_pe and self._reuse_heads < self._num_heads:
            self._position_embeddings = initializer([
                       1, self._num_heads - self._reuse_heads, 2 * self.
                       _pe_max_seq_length - 1],['truncated_normal',0.2],self.dtype)
            self.param.append(self._position_embeddings)
        return    
    
    def _compute_relative_position(self, query_seq_length, key_seq_length):
      position_zero = self._pe_max_seq_length - 1
      # We take the vector position variable and concatenate to form a matrix of
      # relative position encodings. i=0 indicates reltaive position is 0.
      indices = tf.expand_dims(tf.range(0, -query_seq_length, -1),
                               -1) + tf.range(key_seq_length) + position_zero
      indices = tf.maximum(indices, 0)
      indices = tf.minimum(indices, 2*self._pe_max_seq_length-2)
      attention_biases = tf.gather(self._position_embeddings, indices, axis=2)
      return attention_biases
    
    def _compute_attention(self,
                           query,
                           key,
                           value,
                           reuse_scores=None,
                           attention_mask=None,
                           train_flag=True):
      """Applies Dot-product attention with query, key, value tensors.
    
      This function defines the computation inside `call` with projected
      multi-head Q, K, V inputs. Users can override this function for customized
      attention implementation.
    
      Args:
        query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
        key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
        value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
        reuse_scores: Attention scores from a previous layer if needed.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
          attention to certain positions.
        training: Python boolean indicating whether the layer should behave in
          training mode (adding dropout) or in inference mode (doing nothing).
    
      Returns:
        attention_output: Multi-headed outputs of attention computation.
        attention_scores: Multi-headed attention weights.
      """
      # Partial or no reuse
      if self._reuse_heads < self._num_heads:
        query = tf.multiply(query, 1.0 / tf.math.sqrt(float(self._key_dim)))
        new_scores = tf.einsum(self._dot_product_equation, key, query)
        # Add relative position embeddings if required.
        if self._use_relative_pe:
          new_scores = new_scores + self._compute_relative_position(
              tf.shape(query)[1], tf.shape(key)[1])
        new_scores = self._masked_softmax(new_scores, attention_mask)
        if self._reuse_heads > 0:  # Partial reuse
          reuse_scores = reuse_scores[:, :self._reuse_heads, :, :]
          attention_scores = tf.concat([new_scores, reuse_scores], 1)
        else:  # No reuse
          attention_scores = new_scores
      else:  # Full reuse
        attention_scores = reuse_scores
        new_scores = None
    
      # `context_layer` = [B, T, N, H]
      attention_output = []
      # Partial or full reuse
      if self._reuse_heads > 0:
        if train_flag:
            attention_output.append(
                tf.einsum(self._combine_equation, tf.nn.dropout(
                    reuse_scores, self._dropout), value[0]))
        else:
            attention_output.append(
                tf.einsum(self._combine_equation, reuse_scores, value[0]))
      # Partial or no reuse
      if self._reuse_heads < self._num_heads:
        if train_flag:
            attention_output.append(
                tf.einsum(self._combine_equation, tf.nn.dropout(
                    new_scores, self._dropout), value[-1]))
        else:
            attention_output.append(
                tf.einsum(self._combine_equation, new_scores, value[-1]))
      return attention_output, attention_scores
    
    def __call__(self,
             query,
             value,
             key=None,
             attention_mask=None,
             return_attention_scores=False,
             train_flag=True,
             reuse_attention_scores=None):
      if self._reuse_heads > 0 and reuse_attention_scores is None:
        raise ValueError("reuse_attention_scores cannot be None when "
                         "reuse_attention is True or > 0.")
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
          
      #   N = `num_attention_heads`
      #   H = `size_per_head`
      # `value` = [B, S, N, H]
      value = [vd(value) for vd in self.value_dense]
      for i in range(len(value)):
          n_batch, n_ctx, n_state = value[i].shape
          value[i] = tf.reshape(value[i], [n_batch, n_ctx, self.n_head, -1])
      if self._reuse_heads < self._num_heads:
        # `query` = [B, T, N ,H]
        query = self.query_dense(query)
        n_batch, n_ctx, n_state = query.shape
        query = tf.reshape(query, [n_batch, n_ctx, self.n_head, -1])
    
        # `key` = [B, S, N, H]
        key = self.key_dense(key)
        n_batch, n_ctx, n_state = key.shape
        key = tf.reshape(key, [n_batch, n_ctx, self.n_head, -1])
      else:
        query, key = None, None
    
      attention_output, attention_scores = self._compute_attention(
          query, key, value, reuse_attention_scores, attention_mask, train_flag)
      for i in range(len(attention_output)):
          B, T, _, _ = attention_output[i].shape
          attention_output[i] = tf.reshape(attention_output[i], [B, T, -1])
      attention_output = [od(attention_output[i]) for i, od in enumerate(
          self.output_dense)]
      if len(attention_output) == 1:
        attention_output = attention_output[0]
      else:
        attention_output = attention_output[0] + attention_output[1]
    
      if return_attention_scores:
        return attention_output, attention_scores
      return attention_output