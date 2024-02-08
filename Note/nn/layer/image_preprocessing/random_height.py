import tensorflow as tf

H_AXIS = -3
W_AXIS = -2

def method(interpolation):
    if interpolation=='area':
        return tf.image.ResizeMethod.AREA
    elif interpolation=='bicubic':
        return tf.image.ResizeMethod.BICUBIC
    elif interpolation=='bilinear':
        return tf.image.ResizeMethod.BILINEAR
    elif interpolation=='gaussian':
        return tf.image.ResizeMethod.GAUSSIAN
    elif interpolation=='lanczos3':
        return tf.image.ResizeMethod.LANCZOS3
    elif interpolation=='lanczos5':
        return tf.image.ResizeMethod.LANCZOS5
    elif interpolation=='mitchellcubic':
        return tf.image.ResizeMethod.MITCHELLCUBIC
    elif interpolation=='nearest':
        return tf.image.ResizeMethod.NEAREST_NEIGHBOR

class random_height:
    """A preprocessing layer which randomly varies image height during training.

    This layer adjusts the height of a batch of images by a random factor.
    The input should be a 3D (unbatched) or 4D (batched) tensor in the
    `"channels_last"` image data format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype. By
    default, the layer will output floats.


    By default, this layer is inactive during inference.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        factor: A positive float (fraction of original height),
            or a tuple of size 2 representing lower and upper bound
            for resizing vertically. When represented as a single float,
            this value is used for both the upper and
            lower bound. For instance, `factor=(0.2, 0.3)` results
            in an output with
            height changed by a random amount in the range `[20%, 30%]`.
            `factor=(-0.2, 0.3)` results in an output with height
            changed by a random amount in the range `[-20%, +30%]`.
            `factor=0.2` results in an output with
            height changed by a random amount in the range `[-20%, +20%]`.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
            Defaults to `"bilinear"`.
        seed: Integer. Used to create a random seed.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., random_height, width, channels)`.
    """

    def __init__(self, factor, interpolation="bilinear", seed=7):
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.height_lower = factor[0]
            self.height_upper = factor[1]
        else:
            self.height_lower = -factor
            self.height_upper = factor

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`factor` argument cannot have an upper bound lesser than the "
                f"lower bound. Received: factor={factor}"
            )
        if self.height_lower < -1.0 or self.height_upper < -1.0:
            raise ValueError(
                "`factor` argument must have values larger than -1. "
                f"Received: factor={factor}"
            )
        self.interpolation = interpolation
        self._interpolation_method = method(
            interpolation
        )
        self.random_generator = tf.random.Generator.from_seed(seed)

    def __call__(self, data, train_flag=True):
        def random_height_inputs(data):
            """Inputs height-adjusted with random ops."""
            inputs_shape = tf.shape(data)
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = inputs_shape[W_AXIS]
            height_factor = self.random_generator.uniform(
                shape=[],
                minval=(1.0 + self.height_lower),
                maxval=(1.0 + self.height_upper),
            )
            adjusted_height = tf.cast(height_factor * img_hd, tf.int32)
            adjusted_size = tf.stack([adjusted_height, img_wd])
            output = tf.image.resize(
                images=data,
                size=adjusted_size,
                method=self._interpolation_method,
            )
            # tf.resize will output float32 regardless of input type.
            output = tf.cast(output, data.dtype)
            output_shape = data.shape.as_list()
            output_shape[H_AXIS] = None
            output.set_shape(output_shape)
            return output

        if train_flag:
            return random_height_inputs(data)
        else:
            return data