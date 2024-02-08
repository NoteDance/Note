import tensorflow as tf
import numpy as np


class reshape:
    """Layer that reshapes inputs into the given shape.

    Input shape:
      Arbitrary, although all dimensions in the input shape must be known/fixed.
      Use the keyword argument `input_shape` (tuple of integers, does not
      include the samples/batch size axis) when using this layer as the first
      layer in a model.

    Output shape:
      `(batch_size,) + target_shape`
    """

    def __init__(self, target_shape):
        self.target_shape = tuple(target_shape)
        self.output_size=self.target_shape[-1]

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Find and replace a missing dimension in an output shape.

        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

        Args:
          input_shape: Shape of array being reshaped
          output_shape: Desired shape of the array with at most a single -1
            which indicates a dimension that should be derived from the input
            shape.

        Returns:
          The new output shape with a -1 replaced with its computed value.

        Raises:
          ValueError: If the total array size of the output_shape is
          different than the input_shape, or more than one unknown dimension
          is specified.
        """
        output_shape = list(output_shape)
        msg = (
            "total size of new array must be unchanged, "
            "input_shape = {}, output_shape = {}".format(
                input_shape, output_shape
            )
        )

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError(
                        "There must be at most one unknown dimension in "
                        f"output_shape. Received: output_shape={output_shape}."
                    )
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)
        return output_shape

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if None in input_shape[1:]:
            output_shape = [input_shape[0]]
            # input shape (partially) unknown? replace -1's with None's
            output_shape += tuple(
                s if s != -1 else None for s in self.target_shape
            )
        else:
            output_shape = [input_shape[0]]
            output_shape += self._fix_unknown_dimension(
                input_shape[1:], self.target_shape
            )
        return tf.TensorShape(output_shape)

    def __call__(self, data):
        result = tf.reshape(data, (tf.shape(data)[0],) + self.target_shape)
        if not tf.executing_eagerly():
            # Set the static shape for the result since it might lost during
            # array_ops reshape, eg, some `None` dim in the result could be
            # inferred.
            result.set_shape(self.compute_output_shape(data.shape))
        return result