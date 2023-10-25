import tensorflow as tf
from typing import Optional


class self_attention_mask:
  """Create 3D attention mask from a 2D tensor mask.

    inputs[0]: from_tensor: 2D or 3D Tensor of shape
      [batch_size, from_seq_length, ...].
    inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """

  def output(self, inputs, to_mask=None):
    if isinstance(inputs, list) and to_mask is None:
      to_mask = inputs[1]
      inputs = inputs[0]
    return get_mask(inputs, to_mask)


def get_mask(inputs: tf.Tensor,
             to_mask: tf.Tensor,
             dtype: Optional[tf.DType] = None) -> tf.Tensor:
  """Gets a 3D self-attention mask.

  Args:
    inputs: from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length,
      ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    dtype: the output Tensor dtype.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = tf.shape(inputs)
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]
  dtype = inputs.dtype if dtype is None else dtype

  to_shape = tf.shape(to_mask)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=dtype)

  return tf.broadcast_to(to_mask, [batch_size, from_seq_length, to_seq_length])