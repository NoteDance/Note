import tensorflow as tf


class rescaling:
    """A preprocessing layer which rescales input values to a new range.

    This layer rescales every value of an input (often an image) by multiplying
    by `scale` and adding `offset`.

    For instance:

    1. To rescale an input in the `[0, 255]` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.

    2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
    you would pass `scale=1./127.5, offset=-1`.

    The rescaling is applied both during training and inference. Inputs can be
    of integer or floating point dtype, and by default the layer will output
    floats.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Input shape:
        Arbitrary.

    Output shape:
        Same as input.

    Args:
        scale: Float, the scale to apply to the inputs.
        offset: Float, the offset to apply to the inputs.
    """

    def __init__(self, scale, offset=0.0):
        self.scale = scale
        self.offset = offset

    def __call__(self, data):
        dtype = data.dtype
        scale = tf.cast(self.scale, dtype)
        offset = tf.cast(self.offset, dtype)
        return tf.cast(data, dtype) * scale + offset