import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.initializer import initializer_

class masked_lm:
  """Masked language model network head for BERT modeling.

  This layer implements a masked language model based on the provided
  transformer based encoder. It assumes that the encoder network being passed
  has a "get_embedding_table()" method.

  Example:
  ```python
  encoder=modeling.networks.BertEncoder(...)
  lm_layer=MaskedLM(embedding_table=encoder.get_embedding_table())
  ```

  Args:
    activation: The activation, if any, for the dense layer.
    initializer: The initializer for the dense layer. Defaults to a Glorot
      uniform initializer.
    output: The output style for this layer. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               vocab_size,
               hidden_size,
               input_size=None,
               activation=None,
               initializer='Xavier',
               output='logits',
               dtype='float32'
               ):
    self.activation = activation
    self.initializer = initializer

    if output not in ('predictions', 'logits'):
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)
    self._output_type = output
    self._vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.output_size = self._vocab_size
    self.dtype = dtype
    if input_size is not None:
        self.dense = dense(
            hidden_size,
            input_size,
            activation=self.activation,
            weight_initializer=self.initializer,
            dtype=dtype
            )
        self.layer_norm = layer_norm(hidden_size,
            axis=-1, epsilon=1e-12)
        self.bias = initializer_(
            shape=(self._vocab_size,),
            initializer='zeros',
            dtype=dtype
            )

  def build(self):
    self.dense = dense(
        self.hidden_size,
        self.input_size,
        activation=self.activation,
        weight_initializer=self.initializer,
        dtype=self.dtype
        )
    self.layer_norm = layer_norm(self.hidden_size,
        axis=-1, epsilon=1e-12)
    self.bias = initializer_(
        shape=(self._vocab_size,),
        initializer='zeros',
        dtype=self.dtype
        )

  def __call__(self, sequence_data, embedding_table, masked_positions):
    if sequence_data.dtype!=self.dtype:
        sequence_data=tf.cast(sequence_data,self.dtype)
    if embedding_table.dtype!=self.dtype:
        embedding_table=tf.cast(embedding_table,self.dtype)
    if masked_positions.dtype!=self.dtype:
        masked_positions=tf.cast(masked_positions,self.dtype)
    masked_lm_input = self._gather_indexes(sequence_data, masked_positions)
    if self.input_size==None:
        self.input_size=masked_lm_input.shape[-1]
        self.build()
    lm_data = self.dense(masked_lm_input)
    lm_data = self.layer_norm(lm_data)
    lm_data = tf.matmul(lm_data, embedding_table, transpose_b=True)
    logits = lm_data+self.bias
    masked_positions_length = masked_positions.shape.as_list()[1] or tf.shape(
        masked_positions)[1]
    logits = tf.reshape(logits,
                        [-1, masked_positions_length, self._vocab_size])
    if self._output_type == 'logits':
      return logits
    return tf.nn.log_softmax(logits)

  def _gather_indexes(self, sequence_tensor, positions):
    """Gathers the vectors at the specific positions, for performance.

    Args:
        sequence_tensor: Sequence output of shape
          (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
          hidden units.
        positions: Positions ids of tokens in sequence to mask for pretraining
          of with dimension (batch_size, num_predictions) where
          `num_predictions` is maximum number of tokens to mask out and predict
          per each sequence.

    Returns:
        Masked out sequence tensor of shape (batch_size * num_predictions,
        num_hidden).
    """
    sequence_shape = tf.shape(sequence_tensor)
    batch_size, seq_length = sequence_shape[0], sequence_shape[1]
    width = sequence_tensor.shape.as_list()[2] or sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(
        positions + tf.cast(flat_offsets, positions.dtype), [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

    return output_tensor