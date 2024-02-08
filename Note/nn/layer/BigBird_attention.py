import tensorflow as tf
from Note.nn.layer.dense import dense
import numpy as np


MAX_SEQ_LEN = 4096


class BigBird_attention:
  """BigBird, a sparse attention mechanism.

  This layer follows the paper "Big Bird: Transformers for Longer Sequences"
  (https://arxiv.org/abs/2007.14062).
  It reduces this quadratic dependency of attention
  computation to linear.

  Arguments are the same as `MultiHeadAttention` layer.
  """

  def __init__(self,
               n_head, 
               key_dim, 
               input_size=None,
               num_rand_blocks=3,
               from_block_size=64,
               to_block_size=64,
               max_rand_mask_length=MAX_SEQ_LEN,
               weight_initializer='Xavier', 
               bias_initializer='zeros', 
               use_bias=True,
               seed=None,
               dtype='float32'
               ):
    self.n_head=n_head
    self.key_dim=key_dim
    self.input_size=input_size
    self.weight_initializer=weight_initializer
    self.bias_initializer=bias_initializer
    self.use_bias=use_bias
    self.dtype=dtype
    self._num_rand_blocks = num_rand_blocks
    self._from_block_size = from_block_size
    self._to_block_size = to_block_size
    self._seed = seed

    # Generates random attention.
    np.random.seed(self._seed)
    # pylint: disable=g-complex-comprehension
    rand_attn = [
        bigbird_block_rand_mask(
            max_rand_mask_length,
            max_rand_mask_length,
            from_block_size,
            to_block_size,
            num_rand_blocks,
            last_idx=1024) for _ in range(self.n_head)
    ]
    # pylint: enable=g-complex-comprehension
    rand_attn = np.stack(rand_attn, axis=0)
    self.rand_attn = tf.constant(rand_attn, dtype=tf.int32)
    if input_size is not None:
      self.query_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
      self.key_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
      self.value_dense=dense(n_head*key_dim,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
      self.output_dense=dense(input_size,n_head*key_dim,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
      self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param]

  def build(self):
      self.query_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
      self.key_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
      self.value_dense=dense(self.n_head*self.key_dim,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
      self.output_dense=dense(self.input_size,self.n_head*self.key_dim,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
      self.param=[self.query_dense.param,self.key_dense.param,self.value_dense.param,self.output_dense.param]
      return
  
  def _compute_attention(self, query, key, value, attention_mask=None):
    (band_mask, encoder_from_mask, encoder_to_mask,
     blocked_encoder_mask) = attention_mask
    query_shape = tf.shape(query)
    from_seq_length = query_shape[1]
    to_seq_length = tf.shape(key)[1]
    rand_attn = self.rand_attn[:, :(from_seq_length // self._from_block_size -
                                    2)]
    return bigbird_block_sparse_attention(
        query,
        key,
        value,
        band_mask,
        encoder_from_mask,
        encoder_to_mask,
        blocked_encoder_mask,
        blocked_encoder_mask,
        num_attention_heads=self.n_head,
        num_rand_blocks=self._num_rand_blocks,
        size_per_head=self.key_dim,
        batch_size=query_shape[0],
        from_seq_length=from_seq_length,
        to_seq_length=to_seq_length,
        from_block_size=self._from_block_size,
        to_block_size=self._to_block_size,
        rand_attn=rand_attn)

  def __call__(self, query, value, key=None, attention_mask=None):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
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
    # `query` = [B, T, N ,H]
    query = self.query_dense(query)
    n_batch, n_ctx, n_state = query.shape
    query = tf.reshape(query, [n_batch, n_ctx, self.n_head, -1])

    # `key` = [B, S, N, H]
    key = self.key_dense(key)
    n_batch, n_ctx, n_state = key.shape
    key = tf.reshape(key, [n_batch, n_ctx, self.n_head, -1])

    # `value` = [B, S, N, H]
    value = self.value_dense(value)
    n_batch, n_ctx, n_state = value.shape
    value = tf.reshape(value, [n_batch, n_ctx, self.n_head, -1])

    attention_output = self._compute_attention(query, key, value,
                                               attention_mask)
    B, S, _, _ = attention_output.shape
    attention_output = tf.reshape(attention_output, [B, S, -1])
    attention_output = self.output_dense(attention_output)
    return attention_output


def bigbird_block_sparse_attention(
    query_layer, key_layer, value_layer, band_mask, from_mask, to_mask,
    from_blocked_mask, to_blocked_mask, rand_attn, num_attention_heads,
    num_rand_blocks, size_per_head, batch_size, from_seq_length, to_seq_length,
    from_block_size, to_block_size):
  """BigBird attention sparse calculation using blocks in linear time.

  Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.


  Args:
    query_layer: float Tensor of shape [batch_size, num_attention_heads,
      from_seq_length, size_per_head]
    key_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    value_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    band_mask: (optional) int32 Tensor of shape [batch_size, 1,
      from_seq_length//from_block_size-4, from_block_size, 3*to_block_size]. The
      values should be 1 or 0. The attention scores will effectively be set to
      -infinity for any positions in the mask that are 0, and will be unchanged
      for positions that are 1.
    from_mask: (optional) int32 Tensor of shape [batch_size, 1, from_seq_length,
      1]. The values should be 1 or 0. The attention scores will effectively be
      set to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    to_mask: (optional) int32 Tensor of shape [batch_size, 1, 1, to_seq_length].
      The values should be 1 or 0. The attention scores will effectively be set
      to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    from_blocked_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size]. Same as from_mask,
      just reshaped.
    to_blocked_mask: (optional) int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size]. Same as to_mask, just
      reshaped.
    rand_attn: [batch_size, num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks]
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    size_per_head: int. Size of each attention head.
    batch_size: int. Batch size for computation.
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
  """
  rand_attn = tf.expand_dims(rand_attn, 0)
  rand_attn = tf.repeat(rand_attn, batch_size, 0)

  rand_mask = create_rand_mask_from_inputs(
      from_blocked_mask,
      to_blocked_mask,
      rand_attn,
      num_attention_heads,
      num_rand_blocks,
      batch_size,
      from_seq_length,
      from_block_size,
  )

  # Define shorthands
  h = num_attention_heads
  r = num_rand_blocks
  d = size_per_head
  b = batch_size
  m = from_seq_length
  n = to_seq_length
  wm = from_block_size
  wn = to_block_size
  dtype = query_layer.dtype
  query_layer = tf.transpose(query_layer, perm=[0, 2, 1, 3])
  key_layer = tf.transpose(key_layer, perm=[0, 2, 1, 3])
  value_layer = tf.transpose(value_layer, perm=[0, 2, 1, 3])
  blocked_query_matrix = tf.reshape(query_layer, (b, h, m // wm, wm, -1))
  blocked_key_matrix = tf.reshape(key_layer, (b, h, n // wn, wn, -1))
  blocked_value_matrix = tf.reshape(value_layer, (b, h, n // wn, wn, -1))
  gathered_key = tf.reshape(
      tf.gather(blocked_key_matrix, rand_attn, batch_dims=2, name="gather_key"),
      (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]
  gathered_value = tf.reshape(
      tf.gather(
          blocked_value_matrix, rand_attn, batch_dims=2, name="gather_value"),
      (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]
  first_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 0],
      key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
  first_product = tf.multiply(first_product, 1.0 / np.sqrt(d))
  first_product += (1.0 - tf.cast(to_mask, dtype=dtype)) * -10000.0
  first_attn_weights = tf.nn.softmax(first_product)  # [b, h, wm, n]
  first_context_layer = tf.einsum(
      "BHQK,BHKD->BHQD", first_attn_weights,
      value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
  first_context_layer = tf.expand_dims(first_context_layer, 2)

  second_key_mat = tf.concat([
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1],
      blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :,
                                                      -1], gathered_key[:, :, 0]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_value_mat = tf.concat([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1],
      blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, 0]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 1], second_key_mat
  )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
  second_seq_pad = tf.concat([
      to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
      tf.ones([b, 1, 1, r * wn], dtype=dtype)
  ], 3)
  second_rand_pad = tf.concat([
      tf.ones([b, h, wm, 4 * wn], dtype=dtype), rand_mask[:, :, 0]
  ], 3)
  second_product = tf.multiply(second_product, 1.0 / np.sqrt(d))
  second_product += (1.0 -
                     tf.minimum(second_seq_pad, second_rand_pad)) * -10000.0
  second_attn_weights = tf.nn.softmax(second_product)  # [b , h, wm, (4+r)*wn]
  second_context_layer = tf.einsum(
      "BHQK,BHKD->BHQD", second_attn_weights, second_value_mat
  )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
  second_context_layer = tf.expand_dims(second_context_layer, 2)

  exp_blocked_key_matrix = tf.concat([
      blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2],
      blocked_key_matrix[:, :, 3:-1]
  ], 3)  # [b, h, m//wm-4, 3*wn, -1]
  exp_blocked_value_matrix = tf.concat([
      blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2],
      blocked_value_matrix[:, :, 3:-1]
  ], 3)  # [b, h, m//wm-4, 3*wn, -1]
  middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
  inner_band_product = tf.einsum(
      "BHLQD,BHLKD->BHLQK", middle_query_matrix, exp_blocked_key_matrix
  )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
  #     ==> [b, h, m//wm-4, wm, 3*wn]
  inner_band_product = tf.multiply(inner_band_product, 1.0 / np.sqrt(d))
  rand_band_product = tf.einsum(
      "BHLQD,BHLKD->BHLQK", middle_query_matrix,
      gathered_key[:, :,
                   1:-1])  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1]
  #     ==> [b, h, m//wm-4, wm, r*wn]
  rand_band_product = tf.multiply(rand_band_product, 1.0 / np.sqrt(d))
  first_band_product = tf.einsum(
      "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, 0]
  )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
  first_band_product = tf.multiply(first_band_product, 1.0 / np.sqrt(d))
  last_band_product = tf.einsum(
      "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, -1]
  )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
  last_band_product = tf.multiply(last_band_product, 1.0 / np.sqrt(d))
  inner_band_product += (1.0 - band_mask) * -10000.0
  first_band_product += (1.0 -
                         tf.expand_dims(to_mask[:, :, :, :wn], 3)) * -10000.0
  last_band_product += (1.0 -
                        tf.expand_dims(to_mask[:, :, :, -wn:], 3)) * -10000.0
  rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
  band_product = tf.concat([
      first_band_product, inner_band_product, rand_band_product,
      last_band_product
  ], -1)  # [b, h, m//wm-4, wm, (5+r)*wn]
  attn_weights = tf.nn.softmax(band_product)  # [b, h, m//wm-4, wm, (5+r)*wn]
  context_layer = tf.einsum(
      "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :,
                                         wn:4 * wn], exp_blocked_value_matrix
  )  # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
  #     ==> [b, h, m//wm-4, wm, -1]
  context_layer += tf.einsum(
      "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :,
                                         4 * wn:-wn], gathered_value[:, :, 1:-1]
  )  # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1]
  #     ==> [b, h, m//wm-4, wm, -1]
  context_layer += tf.einsum(
      "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, :wn],
      blocked_value_matrix[:, :, 0]
  )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
  context_layer += tf.einsum(
      "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :,
                                        -wn:], blocked_value_matrix[:, :, -1]
  )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]

  second_last_key_mat = tf.concat([
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3],
      blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1],
      gathered_key[:, :, -1]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_last_value_mat = tf.concat([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3],
      blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, -1]
  ], 2)  # [b, h, (4+r)*wn, -1]
  second_last_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -2], second_last_key_mat
  )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
  second_last_seq_pad = tf.concat([
      to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
      tf.ones([b, 1, 1, r * wn], dtype=dtype)
  ], 3)
  second_last_rand_pad = tf.concat(
      [tf.ones([b, h, wm, 4 * wn], dtype=dtype), rand_mask[:, :, -1]], 3)
  second_last_product = tf.multiply(second_last_product, 1.0 / np.sqrt(d))
  second_last_product += (
      1.0 - tf.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
  second_last_attn_weights = tf.nn.softmax(
      second_last_product)  # [b, h, wm, (4+r)*wn]
  second_last_context_layer = tf.einsum(
      "BHQK,BHKD->BHQD", second_last_attn_weights, second_last_value_mat
  )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
  second_last_context_layer = tf.expand_dims(second_last_context_layer, 2)

  last_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -1],
      key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
  last_product = tf.multiply(last_product, 1.0 / np.sqrt(d))
  last_product += (1.0 - to_mask) * -10000.0
  last_attn_weights = tf.nn.softmax(last_product)  # [b, h, wm, n]
  last_context_layer = tf.einsum(
      "BHQK,BHKD->BHQD", last_attn_weights,
      value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
  last_context_layer = tf.expand_dims(last_context_layer, 2)

  context_layer = tf.concat([
      first_context_layer, second_context_layer, context_layer,
      second_last_context_layer, last_context_layer
  ], 2)
  context_layer = tf.reshape(context_layer, (b, h, m, -1)) * from_mask
  context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
  return context_layer


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
  """Create adjacency list of random attention.

  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
      if positive then num_rand_blocks blocks choosen only upto last_idx.

  Returns:
    adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
  """
  assert from_seq_length//from_block_size == to_seq_length//to_block_size, \
      "Error the number of blocks needs to be same!"

  rand_attn = np.zeros(
      (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
  middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
  last = to_seq_length // to_block_size - 1
  if last_idx > (2 * to_block_size):
    last = (last_idx // to_block_size) - 1

  r = num_rand_blocks  # shorthand
  for i in range(1, from_seq_length // from_block_size - 1):
    start = i - 2
    end = i
    if i == 1:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
    elif i == 2:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
    elif i == from_seq_length // from_block_size - 3:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
      # Missing -3: should have been sliced till last-3
    elif i == from_seq_length // from_block_size - 2:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
      # Missing -4: should have been sliced till last-4
    else:
      if start > last:
        start = last
        rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
      elif (end + 1) == last:
        rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
      else:
        rand_attn[i - 1, :] = np.random.permutation(
            np.concatenate((middle_seq[:start], middle_seq[end + 1:last])))[:r]
  return rand_attn


def create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn,
                                 num_attention_heads, num_rand_blocks,
                                 batch_size, from_seq_length, from_block_size):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
    rand_attn: [batch_size, num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks]
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    batch_size: int. Batch size for computation.
    from_seq_length: int. length of from sequence.
    from_block_size: int. size of block in from sequence.

  Returns:
    float Tensor of shape [batch_size, num_attention_heads,
                           from_seq_length//from_block_size-2,
                           from_block_size, num_rand_blocks*to_block_size].
  """
  num_windows = from_seq_length // from_block_size - 2
  rand_mask = tf.reshape(
      tf.gather(to_blocked_mask, rand_attn, batch_dims=1), [
          batch_size, num_attention_heads, num_windows,
          num_rand_blocks * from_block_size
      ])
  rand_mask = tf.einsum("BLQ,BHLK->BHLQK", from_blocked_mask[:, 1:-1],
                        rand_mask)
  return rand_mask