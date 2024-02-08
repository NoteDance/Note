import tensorflow as tf


class unit_norm:
    """Unit normalization layer.

    Normalize a batch of inputs so that each input in the batch has a L2 norm
    equal to 1 (across the axes specified in `axis`).

    Example:

    >>> data = tf.constant(np.arange(6).reshape(2, 3), dtype=tf.float32)
    >>> normalized_data = Note.nn.layer.unit_normalization.unit_normalization().output(data)
    >>> print(tf.reduce_sum(normalized_data[0, :] ** 2).numpy())
    1.0

    Args:
      axis: Integer or list/tuple. The axis or axes to normalize across.
        Typically this is the features axis or axes. The left-out axes are
        typically the batch axis or axes. Defaults to `-1`, the last dimension
        in the input.
    """

    def __init__(self, axis=-1):
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Invalid value for `axis` argument: "
                "expected an int or a list/tuple of ints. "
                f"Received: axis={axis}"
            )


    def __call__(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=self.axis)