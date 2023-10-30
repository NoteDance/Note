import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.softmax import softmax
import numpy as np
import collections
import string


_CHR_IDX = string.ascii_lowercase


class cached_attention:
  """Attention layer with cache used for autoregressive decoding.

  Arguments are the same as `tf.keras.layers.MultiHeadAttention` layer.
  """
  def __init__(self,n_head,key_dim,value_dim=None,input_size=None,attention_axes=None,dropout_rate=0.0,weight_initializer="Xavier",bias_initializer="zeros",use_bias=True,dtype='float32'):
      self.n_head=n_head
      self.key_dim=key_dim
      self.value_dim=value_dim if value_dim else key_dim
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
          self.value_dense=dense(n_head*value_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.output_dense=dense(input_size,n_head*value_dim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
          self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param]
  
  def build(self):
    if self.input_size is not None:
        self.query_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.key_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.value_dense=dense(self.n_head*self.value_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.output_dense=dense(self.input_size,self.n_head*self.value_dim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param]
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

  def _update_cache(self, key, value, cache, decode_loop_step):
    """Updates cache states and gets full-length key/value tensors."""
    # Combines cached keys and values with new keys and values.
    if decode_loop_step is not None:
      # TPU special case.
      key_seq_dim = cache["key"].shape.as_list()[1]
      indices = tf.reshape(
          tf.one_hot(decode_loop_step, key_seq_dim, dtype=key.dtype),
          [1, key_seq_dim, 1, 1])
      key = cache["key"] + key * indices
      value_seq_dim = cache["value"].shape.as_list()[1]
      indices = tf.reshape(
          tf.one_hot(decode_loop_step, value_seq_dim, dtype=value.dtype),
          [1, value_seq_dim, 1, 1])
      value = cache["value"] + value * indices
    else:
      key = tf.concat([tf.cast(cache["key"], key.dtype), key], axis=1)
      value = tf.concat([tf.cast(cache["value"], value.dtype), value], axis=1)

    # Update cache
    cache["key"] = key
    cache["value"] = value

    return key, value

  def output(self,
           query,
           value,
           key=None,
           attention_mask=None,
           cache=None,
           decode_loop_step=None,
           return_attention_scores=False):
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
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, F, N ,H]
    query = self.query_dense.output(query)
    n_batch, n_ctx, n_state = query.shape
    query = tf.reshape(query, [n_batch, n_ctx, self.n_head, -1])

    # `key` = [B, T, N, H]
    key = self.key_dense.output(key)
    n_batch, n_ctx, n_state = key.shape
    key = tf.reshape(key, [n_batch, n_ctx, self.n_head, -1])

    # `value` = [B, T, N, H]
    value = self.value_dense.output(value)
    n_batch, n_ctx, n_state = key.shape
    value = tf.reshape(value, [n_batch, n_ctx, self.n_head, -1])

    if cache:
      key, value = self._update_cache(key, value, cache, decode_loop_step)

    query = tf.multiply(query, 1.0 / tf.math.sqrt(float(self._key_dim)))

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum(self._dot_product_equation, key, query)

    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, F, T]
    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores = tf.nn.dropout(attention_scores, self.dropout_rate)
    # `context_layer` = [B, F, N, H]
    attention_output = tf.einsum(self._combine_equation, attention_scores,
                                 value)
    B, F, _, _ = attention_output.shape
    attention_output = tf.reshape(attention_output, [B, F, -1])
    attention_output = self.output_dense.output(attention_output)
    if return_attention_scores:
      return attention_output, attention_scores, cache
    return attention_output, cache

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