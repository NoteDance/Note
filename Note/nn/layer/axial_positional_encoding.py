import tensorflow as tf
from Note import nn


class axial_positional_encoding:
  """A class for generating axial positional encoding for Reformer models."""

  def __init__(self, d_model, axial_shape, initializer='Xavier', trainable=True, dtype='float32'):
    """Initializes the axial positional encoding.

    Args:
      d_model: int, the dimension of the model embeddings.
      axial_shape: tuple of int, the shape of the input sequence, such as (batch_size, seq_length).
    """
    self.d_model = d_model
    self.axial_shape = axial_shape
    self.num_axial_pos_embs = len(axial_shape)
    self.d_axial_pos_embs = d_model // self.num_axial_pos_embs

    # Create the learnable parameters for each axial dimension
    self.weights = []
    
    self.output_size = d_model
    
    self.dtype=dtype
    
    # Create a list to store the parameters
    self.param = []
    
    if trainable==True:
        for i, dim in enumerate(axial_shape):
          weight = nn.initializer((dim, self.d_axial_pos_embs), initializer, dtype)
          self.weights.append(weight)
          self.param.append(weight)
    

  def __call__(self, data):
    """Generates the axial positional encoding for the input tensor.

    Args:
      data: tf.Tensor of shape [batch_size, seq_length, d_model], the input tensor.

    Returns:
      tf.Tensor of shape [batch_size, seq_length, d_model], the output tensor with axial positional encoding added.
    """
    if data.dtype!=self.dtype:
        data=tf.cast(data,self.dtype)
        
    # Reshape the input tensor to match the axial shape
    data = tf.reshape(data, (-1,) + self.axial_shape + (self.d_model,))
    
    # Concatenate the positional embeddings along the last dimension
    pos_emb = tf.concat(
        [tf.expand_dims(weight, axis=0) for weight in self.weights],
        axis=-1
    )

    # Broadcast the positional embeddings to the input shape
    pos_emb = tf.broadcast_to(pos_emb, data.shape)

    # Add the positional embeddings to the input tensor
    data = data + pos_emb

    # Reshape the output tensor to the original shape
    output = tf.reshape(data, (-1, self.axial_shape[0] * self.axial_shape[1], self.d_model))

    return output