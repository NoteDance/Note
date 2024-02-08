import tensorflow as tf


def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    Args:
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    Returns:
        A tensor.

    Example:

        >>> b = tf.constant([[1, 2], [3, 4]])
        >>> b
        <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
        array([[1, 2],
               [3, 4]], dtype=int32)>
        >>> tf.keras.backend.repeat(b, n=2)
        <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
        array([[[1, 2],
                [1, 2]],
               [[3, 4],
                [3, 4]]], dtype=int32)>

    """
    x = tf.expand_dims(x, 1)
    pattern = tf.stack([1, n, 1])
    return tf.tile(x, pattern)


class repeat_vector:
    """Repeats the input n times.

    Args:
      n: Integer, repetition factor.
    Input shape: 2D tensor of shape `(num_samples, features)`.
    Output shape: 3D tensor of shape `(num_samples, n, features)`.
    """

    def __init__(self, n):
        self.n = n
        if not isinstance(n, int):
            raise TypeError(
                f"Expected an integer value for `n`, got {type(n)}."
            )

    def __call__(self, data):
        return repeat(data, self.n)