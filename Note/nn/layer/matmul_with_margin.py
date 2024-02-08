import tensorflow as tf
from typing import Tuple

class matmul_with_margin:
  """This layer computs a dot product matrix given two encoded inputs.

  Args:
    logit_scale: The scaling factor of dot products when doing training.
    logit_margin: The margin value between the positive and negative examples
      when doing training.
  """

  def __init__(self,
               logit_scale=1.0,
               logit_margin=0.0,
               ):
    self.logit_scale = logit_scale
    self.logit_margin = logit_margin

  def __call__(self, left_encoded: tf.Tensor,
           right_encoded: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_size = left_encoded[0]

    # Left -> Right dot product.
    left_dot_products = tf.matmul(
        left_encoded, right_encoded, transpose_b=True)

    self.left_logits = self.logit_scale * (
        left_dot_products - self.logit_margin * tf.eye(batch_size))

    # Right -> Left dot product.
    self.right_logits = tf.transpose(self.left_logits)

    return (self.left_logits, self.right_logits)