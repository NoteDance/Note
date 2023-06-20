import tensorflow as tf
from Note.nn.initializer import initializer

class Ripple_attention:
  def __init__(self, d_model, num_heads, weight_initializer='Xavier', dtype='float32'):
    # check if the hidden size is divisible by the number of heads
    # store the parameters
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_head = d_model // num_heads
    # create trainable variables for query, key and value projections
    self.wq = initializer([d_model, d_model], weight_initializer, dtype=dtype)
    self.wk = initializer([d_model, d_model], weight_initializer, dtype=dtype)
    self.wv = initializer([d_model, d_model], weight_initializer, dtype=dtype)
    # create trainable variables for output projection
    self.wo = initializer([d_model, d_model], weight_initializer, dtype=dtype)
    self.param=[self.wq, self.wk, self.wv, self.wo]

  def split_heads(self, x):
    # split the last dimension into (num_heads, d_head)
    x = tf.reshape(x, [x.shape[0], x.shape[1], self.num_heads, self.d_head])
    # transpose the result such that the shape is (batch_size, num_heads, seq_len, d_head)
    x = tf.transpose(x, [0, 2, 1, 3])
    return x

  def ripple(self, qk):
    # compute the cumulative sum of qk along the sequence dimension
    qk_cumsum = tf.cumsum(qk, axis=2)
    # compute the ripple attention score as qk / sqrt(qk_cumsum + 1e-6)
    score = qk / tf.sqrt(qk_cumsum + 1e-6)
    return score

  def output(self, data1, data2=None):
    # compute the query projection from data1
    q = tf.matmul(data1, self.wq)
    if data2 is not None:
      # if data2 is given, compute the key and value projections from data2
      k = tf.matmul(data2, self.wk)
      v = tf.matmul(data2, self.wv)
    else:
      # if data2 is not given, compute the key and value projections from data1
      k = tf.matmul(data1, self.wk)
      v = tf.matmul(data1, self.wv)
    # split the heads
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)
    # compute the scaled dot product of query and key
    qk = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_head,data1.dtype.name))
    # apply the ripple function to get the attention score
    score = self.ripple(qk)
    # apply softmax to get the attention weight
    weight = tf.nn.softmax(score, axis=-1)
    # compute the weighted sum of value
    out = tf.matmul(weight, v)
    # concatenate the heads
    out = tf.transpose(out, [0, 2, 1, 3])
    out = tf.reshape(out, [out.shape[0], out.shape[1], self.d_model])
    # apply the output projection
    out = tf.matmul(out, self.wo)
    return out
