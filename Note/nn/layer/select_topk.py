import tensorflow as tf


class select_topk:
  """Select top-k + random-k tokens according to importance."""

  def __init__(self,
               top_k=None,
               random_k=None,
               ):
    self._top_k = top_k
    self._random_k = random_k


  def __call__(self, data):
    if self._random_k is None:
      # Pure top-k, not randomness.
      pos = tf.argsort(data, direction="DESCENDING")
      selected = tf.slice(pos, [0, 0], [-1, self._top_k])
      not_selected = tf.slice(pos, [0, self._top_k], [-1, -1])
    elif self._top_k is None:
      # Pure randomness, no top-k.
      pos = tf.argsort(tf.random.uniform(shape=tf.shape(data)),
                       direction="DESCENDING")
      selected = tf.slice(pos, [0, 0], [-1, self._random_k])
      not_selected = tf.slice(pos, [0, self._random_k], [-1, -1])
    else:
      # Top-k plus randomness.
      pos = tf.argsort(data, direction="DESCENDING")
      selected_top_k = tf.slice(pos, [0, 0], [-1, self._top_k])
      pos_left = tf.slice(pos, [0, self._top_k], [-1, -1])

      # Randomly shuffle pos_left
      sort_index = tf.argsort(
          tf.random.uniform(shape=tf.shape(pos_left)),
          direction="DESCENDING")
      pos_left = tf.gather(pos_left, sort_index, batch_dims=1, axis=1)

      selected_rand = tf.slice(pos_left, [0, 0], [-1, self._random_k])
      not_selected = tf.slice(pos_left, [0, self._random_k], [-1, -1])

      selected = tf.concat([selected_top_k, selected_rand], axis=1)

    # Return the indices of selected and not-selected tokens.
    return selected, not_selected