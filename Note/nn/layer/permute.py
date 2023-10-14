import tensorflow as tf


class permute:
    """Permutes the dimensions of the input according to a given pattern.

    Useful e.g. connecting RNNs and convnets.

    Args:
      dims: Tuple of integers. Permutation pattern does not include the
        samples dimension. Indexing starts at 1.
        For instance, `(2, 1)` permutes the first and second dimensions
        of the input.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same as the input shape, but with the dimensions re-ordered according
      to the specified pattern.
    """

    def __init__(self, dims):
        self.dims = tuple(dims)
        if sorted(dims) != list(range(1, len(dims) + 1)):
            raise ValueError(
                "Invalid permutation argument `dims` for Permute Layer. "
                "The set of indices in `dims` must be consecutive and start "
                f"from 1. Received dims={dims}"
            )


    def output(self, data):
        return tf.transpose(data, perm=(0,) + self.dims)