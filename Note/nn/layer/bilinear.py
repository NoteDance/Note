import tensorflow as tf
from Note.nn.initializer import initializer
from typing import Tuple

class bilinear:
  def __init__(self, embedding_dim: int, output_dim: int, dtype='float32'):
    """Initializer.

    Args:
      embedding_dim: An integer that indicates the embedding dimension of the
        interacting vectors.
      output_dim: An integer that indicates the output dimension of the layer.
    """
    self._embedding_dim = embedding_dim
    self._output_dim = output_dim
    self.dtype = dtype
    self._bilinear_weight = initializer(
        shape=(self._embedding_dim, self._embedding_dim, self._output_dim),
        initializer=['normal', 0.0, 1. /self._embedding_dim],
        dtype=dtype)
    self._linear_weight_1 = initializer(
        shape=(self._embedding_dim, self._output_dim),
        initializer=['normal', 0.0, 1. / tf.math.sqrt(self._embedding_dim)],
        dtype=dtype)
    self._linear_weight_2 = initializer(
        shape=(self._embedding_dim, self._output_dim),
        initializer=['normal', 0.0, 1. / tf.math.sqrt(self._embedding_dim)],
        dtype=dtype)
    self._bias = initializer(
        shape=(self._output_dim),
        initializer='zeros',
        dtype=dtype)

  def __call__(self, data: Tuple[tf.Tensor]) -> tf.Tensor:
    """Computes bilinear interaction between two vector tensors.

    Args:
      data: A pair of tensors of the same shape [batch_size, embedding_dim].

    Returns:
      A tensor, of shape [batch_size, output_dim], computed by the bilinear
      interaction.
    """
    # Input of the function must be a list of two tensors.
    vec_1, vec_2 = data
    if vec_1.dtype!=self.dtype:
        vec_1=tf.cast(vec_1, self.dtype)
    if vec_2.dtype!=self.dtype:
        vec_2=tf.cast(vec_2, self.dtype)
    return tf.einsum(
        'bi,ijk,bj->bk', vec_1, self._bilinear_weight, vec_2) + tf.einsum(
            'bi,ik->bk', vec_1, self._linear_weight_1) + tf.einsum(
                'bi,ik->bk', vec_2, self._linear_weight_2) + self._bias