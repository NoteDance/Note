import tensorflow as tf

class BigBird_masks:
  """Creates bigbird attention masks."""

  def __init__(self, block_size):
    self._block_size = block_size

  def output(self, inputs, mask):
    encoder_shape = tf.shape(mask)
    mask = tf.cast(mask, inputs.dtype)
    batch_size, seq_length = encoder_shape[0], encoder_shape[1]
    # reshape for blocking
    blocked_encoder_mask = tf.reshape(
        mask, (batch_size, seq_length // self._block_size, self._block_size))
    encoder_from_mask = tf.reshape(mask, (batch_size, 1, seq_length, 1))
    encoder_to_mask = tf.reshape(mask, (batch_size, 1, 1, seq_length))

    band_mask = create_band_mask_from_inputs(blocked_encoder_mask,
                                             blocked_encoder_mask)
    return [band_mask, encoder_from_mask, encoder_to_mask, blocked_encoder_mask]

def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].

  Returns:
    float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                           from_block_size,  3*to_block_size].
  """
  exp_blocked_to_pad = tf.concat([
      to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:,
                                                                          3:-1]
  ], 2)
  band_mask = tf.einsum("BLQ,BLK->BLQK", from_blocked_mask[:, 2:-2],
                        exp_blocked_to_pad)
  band_mask = tf.expand_dims(band_mask, 1)
  return band_mask