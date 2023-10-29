import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.softmax import softmax
import numpy as np
import collections
import string


_CHR_IDX = string.ascii_lowercase


class multiheadrelative_attention:
    """A multi-head attention layer with relative attention + position encoding.
    
    This layer shares the same input/output projections as the common
    `tf.keras.layers.MultiHeadAttention` layer.
    
    When it calculates attention logits, position encoding is projected to form
    relative keys. The logits are composed by shifted relative logits and content
    logits.
    
    **Note: This layer is currently experimental.
    
    Attributes:
      kernel_initializer: The kernel initializer. Defaults to variance_scaling.
    
    Call args:
      query: Query `Tensor` of shape `[B, T, dim]`.
      value: Value `Tensor` of shape `[B, S, dim]`.
      content_attention_bias: Bias `Tensor` for content based attention of shape
        `[num_heads, dim]`.
      positional_attention_bias: Bias `Tensor` for position based attention of
        shape `[num_heads, dim]`.
      key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      relative_position_encoding: Relative positional encoding `Tensor` of shape
        `[B, L, dim]`.
      segment_matrix: Optional `Tensor` representing segmentation IDs used in
        XLNet of shape `[B, S, S + M]`.
      segment_encoding: Optional `Tensor` representing the segmentation encoding
        as used in XLNet of shape `[2, num_heads, dim]`.
      segment_attention_bias: Optional trainable bias parameter added to the query
        had when calculating the segment-based attention score used in XLNet of
        shape `[num_heads, dim]`.
      state: Optional `Tensor` of shape `[B, M, E]` where M is the length of the
        state or memory. If passed, this is also attended over as in Transformer
        XL.
      attention_mask: A boolean mask of shape `[B, T, S]` that prevents attention
        to certain positions.
    """
    
    def __init__(self, n_head, key_dim, input_size=None, attention_axes=None, dropout_rate=0.0, 
                 weight_initializer=['VarianceScaling',1.0,'fan_in','truncated_normal'], 
                 bias_initializer='zeros', 
                 use_bias=True, 
                 dtype='float32'):
        self.n_head=n_head
        self.key_dim=key_dim
        self.input_size=input_size
        if attention_axes is not None and not isinstance(
            attention_axes, collections.abc.Sized
        ):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, 3 - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(3, attn_axes=self._attention_axes)
        norm_axes = tuple(
            range(
                attn_scores_rank - len(self._attention_axes), attn_scores_rank
            )
        )
        self._softmax = softmax(
                axis=norm_axes
            )
        self.dropout_rate=dropout_rate
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.use_bias=use_bias
        self.dtype=dtype
        if input_size is not None:
            self.query_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.key_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.value_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.output_dense=dense(input_size,n_head*key_dim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.encoding_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
            self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param,self.encoding_dense.param]
    
    
    def build(self):
        self.query_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.key_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.value_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.output_dense=dense(self.input_size,self.n_head*self.key_dim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.encoding_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param,self.encoding_dense.param]
        return
    
    
    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax.output(attention_scores, attention_mask)
    
    
    def compute_attention(self,
                          query,
                          key,
                          value,
                          position,
                          content_attention_bias,
                          positional_attention_bias,
                          segment_matrix=None,
                          segment_encoding=None,
                          segment_attention_bias=None,
                          attention_mask=None):
      """Computes the attention.
    
      This function defines the computation inside `call` with projected
      multihead Q, K, V, R inputs.
    
      Args:
        query: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
        key: Projected key `Tensor` of shape `[B, S + M, N, key_dim]`.
        value: Projected value `Tensor` of shape `[B, S + M, N, key_dim]`.
        position: Projected position `Tensor` of shape `[B, L, N, key_dim]`.
        content_attention_bias: Trainable bias parameter added to the query head
          when calculating the content-based attention score.
        positional_attention_bias: Trainable bias parameter added to the query
          head when calculating the position-based attention score.
        segment_matrix: Optional `Tensor` representing segmentation IDs used in
          XLNet.
        segment_encoding: Optional trainable `Tensor` representing the
          segmentation encoding as used in XLNet.
        segment_attention_bias: Optional trainable bias parameter added to the
          query had when calculating the segment-based attention score used in
          XLNet.
        attention_mask: (default None) Optional mask that is added to attention
          logits. If state is not None, the mask source sequence dimension should
          extend M.
    
      Returns:
        attention_output: Multi-headed output of attention computation of shape
          `[B, S, N, key_dim]`.
    
      """
      content_attention = tf.einsum(self._dot_product_equation,
                                    key,
                                    query + content_attention_bias)
      positional_attention = tf.einsum(self._dot_product_equation,
                                       position,
                                       query + positional_attention_bias)
      positional_attention = _rel_shift(
          positional_attention, klen=tf.shape(content_attention)[3])
    
      if segment_matrix is not None:
        segment_attention = tf.einsum("bind,snd->bnis",
                                      query + segment_attention_bias,
                                      segment_encoding)
        target_shape = tf.shape(positional_attention)
        segment_attention = tf.where(
            tf.broadcast_to(tf.expand_dims(segment_matrix, 1), target_shape),
            tf.broadcast_to(segment_attention[:, :, :, 1:], target_shape),
            tf.broadcast_to(segment_attention[:, :, :, :1], target_shape))
        attention_sum = (
            content_attention + positional_attention + segment_attention)
      else:
        attention_sum = content_attention + positional_attention
    
      attention_scores = tf.multiply(
          attention_sum, 1.0 / tf.math.sqrt(float(self._key_dim)))
    
      attention_scores = self._masked_softmax(attention_scores, attention_mask)
    
      attention_output = tf.nn.dropout(attention_scores, self.dropout_rate)
    
      attention_output = tf.einsum(self._combine_equation,
                                   attention_output,
                                   value)
      return attention_output
    
    
    def output(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
             query,
             value,
             content_attention_bias,
             positional_attention_bias,
             key=None,
             relative_position_encoding=None,
             segment_matrix=None,
             segment_encoding=None,
             segment_attention_bias=None,
             state=None,
             attention_mask=None):
      """Compute multi-head relative attention over inputs.
    
      Size glossary:
        * Number of heads (H): the number of attention heads.
        * Value size (V): the size of each value embedding per head.
        * Key size (K): the size of each key embedding per head. Equally, the size
          of each query embedding per head. Typically K <= V.
        * Batch dimensions (B).
        * Query (target) attention axes shape (T).
        * Value (source) attention axes shape (S), the rank must match the target.
        * Encoding length (L): The relative positional encoding length.
    
      Args:
        query: attention input.
        value: attention input.
        content_attention_bias: A trainable bias parameter added to the query head
          when calculating the content-based attention score.
        positional_attention_bias: A trainable bias parameter added to the query
          head when calculating the position-based attention score.
        key: attention input.
        relative_position_encoding: relative positional encoding for key and
          value.
        segment_matrix: Optional `Tensor` representing segmentation IDs used in
          XLNet.
        segment_encoding: Optional `Tensor` representing the segmentation encoding
          as used in XLNet.
        segment_attention_bias: Optional trainable bias parameter added to the
          query had when calculating the segment-based attention score used in
          XLNet.
        state: (default None) optional state. If passed, this is also attended
          over as in TransformerXL.
        attention_mask: (default None) Optional mask that is added to attention
          logits. If state is not None, the mask source sequence dimension should
          extend M.
    
      Returns:
        attention_output: The result of the computation, of shape [B, T, E],
          where `T` is for target sequence shapes and `E` is the query input last
          dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
          are projected to the shape specified by `output_shape`.
      """
      if key is None:
        key = value
      if state is not None and state.shape.ndims > 1:
        value = tf.concat([state, value], 1)
        key = tf.concat([state, key], 1)
            
      if query.dtype!=self.dtype:
          query=tf.cast(query,self.dtype)
      if value.dtype!=self.dtype:
          value=tf.cast(value,self.dtype)
      if key.dtype!=self.dtype:
          key=tf.cast(key,self.dtype)
      if relative_position_encoding.dtype!=self.dtype:
          relative_position_encoding=tf.cast(relative_position_encoding,self.dtype)
    
      if self.input_size==None:
          self.input_size=query.shape[-1]
          self.build()
    
      # `query` = [B, T, N ,H]
      query = self.query_dense.output(query)
      n_batch, n_ctx, n_state = query.shape
      query = tf.reshape(query, [n_batch, n_ctx, self.n_head, -1])
    
      # `key` = [B, S + M, N, H]
      key = self.key_dense.output(key)
      n_batch, n_ctx, n_state = key.shape
      key = tf.reshape(key, [n_batch, n_ctx, self.n_head, -1])
    
      # `value` = [B, S + M, N, H]
      value = self.value_dense.output(value)
      n_batch, n_ctx, n_state = value.shape
      value = tf.reshape(value, [n_batch, n_ctx, self.n_head, -1])
    
      # `position` = [B, L, N, H]
      position = self.encoding_dense.output(relative_position_encoding)
      n_batch, n_ctx, n_state = position.shape
      position = tf.reshape(position, [n_batch, n_ctx, self.n_head, -1])
    
      attention_output = self.compute_attention(
          query=query,
          key=key,
          value=value,
          position=position,
          content_attention_bias=content_attention_bias,
          positional_attention_bias=positional_attention_bias,
          segment_matrix=segment_matrix,
          segment_encoding=segment_encoding,
          segment_attention_bias=segment_attention_bias,
          attention_mask=attention_mask)
    
      # `attention_output` = [B, S, N, H]
      B, S, _, _ = attention_output.shape
      attention_output = tf.reshape(attention_output, [B, S, -1])
      attention_output = self.output_dense.output(attention_output)
    
      return attention_output


def _rel_shift(x, klen=-1):
  """Performs relative shift to form the relative attention score."""

  x = tf.transpose(x, perm=[2, 3, 0, 1])
  x_size = tf.shape(x)

  x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
  x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
  x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

  x = tf.transpose(x, perm=[2, 3, 0, 1])

  return x


def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
    dims>, <query attention dims>, num_heads, channels)`

    Args:
        rank: Rank of query, key, value tensors.
        attn_axes: List/tuple of axes, `[-1, rank)`,
            that attention will be applied to.

    Returns:
        Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join(
        [target_notation[i] for i in batch_dims]
        + [target_notation[i] for i in attn_axes]
        + [source_notation[i] for i in attn_axes]
    )
    dot_product_equation = "%s,%s->%s" % (
        source_notation,
        target_notation,
        product_notation,
    )
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (
        product_notation,
        source_notation,
        target_notation,
    )
    return dot_product_equation, combine_equation, attn_scores_rank
