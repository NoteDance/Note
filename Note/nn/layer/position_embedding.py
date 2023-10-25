import tensorflow as tf
from Note.nn.initializer import initializer_


class position_embedding:
  """Creates a positional embedding.

  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    seq_axis: The axis of the input tensor where we add the embeddings.

  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
  """

  def __init__(self,
               max_length,
               input_size=None,
               initializer="Xavier",
               seq_axis=1,
               dtype='float32'
               ):

    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self.max_length = max_length
    self.input_size = input_size
    self.initializer = initializer
    self._seq_axis = seq_axis
    self.dtype = dtype
    if input_size is not None:
        self._position_embeddings = initializer_([max_length, input_size], initializer, dtype)
        self.param=[self._position_embeddings]

  
  def build(self):
      self._position_embeddings = initializer_([self.max_length, self.input_size], self.initializer, self.dtype)
      self.param=[self._position_embeddings]
      return


  def output(self, data):
    input_shape = tf.shape(data)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]
    new_shape = [1 for _ in data.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)