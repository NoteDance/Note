import tensorflow as tf
from Note.nn.initializer import initializer_
import math
import string
import collections


_CHR_IDX = string.ascii_lowercase


class talking_heads_attention:
    def __init__(self,attention_axes=None,dropout_rate=0.0,initializer='Xavier',dtype='float32'):
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
        self.dropout_rate=dropout_rate
        num_batch_dims = 3 - len(self._attention_axes) - 2
    
        # The shape of attn_scores is:
        # (<batch_dims>, num_heads, <query_attn_dims>, <key_attn_dims>)
        attn_scores_rank = num_batch_dims + 1 + len(self._attention_axes) * 2
        scores_notation = _CHR_IDX[:attn_scores_rank]
        projection_notation = scores_notation[num_batch_dims] + (
            _CHR_IDX[attn_scores_rank])
        projected_scores_notation = scores_notation[:num_batch_dims] + (
            _CHR_IDX[attn_scores_rank] + scores_notation[num_batch_dims + 1:])
        self._talking_heads_equation = "%s,%s->%s" % (
            scores_notation, projection_notation, projected_scores_notation)
    
        self._pre_softmax_weight = initializer_((self._num_heads, self._num_heads), initializer, dtype)
        self._post_softmax_weight = initializer_((self._num_heads, self._num_heads), initializer, dtype)
        self.param=[self._pre_softmax_weight, self._post_softmax_weight]


    def output(self,
              query_tensor,
              key_tensor,
              value_tensor,
              attention_mask=None,
              train_flag=True):
          """Applies Dot-product attention with query, key, value tensors.
      
          This function overrides base class to apply additional linear projection
          on attention scores before and after softmax.
      
          Args:
            query_tensor: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
            key_tensor: Projected key `Tensor` of shape `[B, T, N, key_dim]`.
            value_tensor: Projected value `Tensor` of shape `[B, T, N, value_dim]`.
            attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
              attention to certain positions.
            train_flag: Python boolean indicating whether the layer should behave in
              training mode (adding dropout) or in inference mode (doing nothing).
      
          Returns:
            attention_output: Multi-headed outputs of attention computation.
            attention_scores: Multi-headed attention weights.
          """
          # Take the dot product between "query" and "key" to get the raw
          # attention scores.
          attention_scores = tf.einsum(self._dot_product_equation, key_tensor,
                                       query_tensor)
          attention_scores = tf.multiply(attention_scores,
                                         1.0 / math.sqrt(float(self._key_dim)))
      
          # Apply linear projection before softmax
          attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
                                       self._pre_softmax_weight)
      
          # Normalize the attention scores to probabilities.
          # `attention_scores` = [B, N, T, S]
          attention_scores = self._masked_softmax(attention_scores, attention_mask)
      
          # Apply linear projection after softmax
          attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
                                       self._post_softmax_weight)
      
          # This is actually dropping out entire tokens to attend to, which might
          # seem a bit unusual, but is taken from the original Transformer paper.
          if train_flag:
              attention_scores_dropout = tf.nn.dropout(
                  attention_scores, self.dropout_rate)
      
          # `context_layer` = [B, T, N, H]
          attention_output = tf.einsum(self._combine_equation,
                                       attention_scores_dropout, value_tensor)
          return attention_output, attention_scores