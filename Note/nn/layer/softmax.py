import tensorflow as tf


def _large_compatible_negative(tensor_type):
    """Large negative number as Tensor.

    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16

    Args:
        tensor_type: a dtype to determine the type.

    Returns:
        a large negative number.
    """
    # In case of dtype=float16 (e.g., for mixed-precision), the largest
    # negative number (dtypes.float16.min) is divided by 2, in order to
    # avoid overflows when summing negative inputs.
    if tensor_type == tf.float16:
        return tf.float16.min / 2.0
    return -1e9


class softmax:
    """Softmax activation function.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
    Call arguments:
        inputs: The inputs, or logits to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.


    Returns:
        Softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1):
        self.axis = axis


    def output(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype)
            )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(
                    inputs
                    - tf.reduce_logsumexp(inputs, axis=self.axis, keepdims=True)
                )
            else:
                return tf.nn.softmax(inputs, axis=self.axis[0])
        return tf.nn.softmax(inputs, axis=self.axis)