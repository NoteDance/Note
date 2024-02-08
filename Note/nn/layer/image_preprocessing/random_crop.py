import tensorflow as tf

H_AXIS = -3
W_AXIS = -2

class random_crop:
    """A preprocessing layer which randomly crops images during training.

    During training, this layer will randomly choose a location to crop images
    down to a target size. The layer will crop all the images in the same batch
    to the same cropping location.

    At inference time, and during training if an input image is smaller than the
    target size, the input will be resized and cropped so as to return the
    largest possible window in the image that matches the target aspect ratio.
    If you need to apply random cropping at inference time, set `training` to
    True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`.

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        seed: Integer. Used to create a random seed.
    """

    def __init__(self, height, width, seed=7):
        self.height = height
        self.width = width
        self.seed = seed
        self.random_generator = tf.random.Generator.from_seed(seed)

    def __call__(self, data, train_flag=True):
        input_shape = tf.shape(data)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width

        def random_crop():
            dtype = input_shape.dtype
            rands = self.random_generator.uniform(
                [2], 0, dtype.max, dtype
            )
            h_start = rands[0] % (h_diff + 1)
            w_start = rands[1] % (w_diff + 1)
            return tf.image.crop_to_bounding_box(
                data, h_start, w_start, self.height, self.width
            )

        def resize():
            outputs = tf.image.resize(data, [self.height, self.width], method=tf.image.ResizeMethod.BILINEAR )
            # smart_resize will always output float32, so we need to re-cast.
            return tf.cast(outputs, data.dtype)

        return tf.cond(
            tf.reduce_all((train_flag, h_diff >= 0, w_diff >= 0)),
            random_crop,
            resize,
        )