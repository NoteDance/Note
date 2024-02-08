import tensorflow as tf


class random_brightness:
    """A preprocessing layer which randomly adjusts brightness during training.

    This layer will randomly increase/reduce the brightness for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    Note that different brightness adjustment factors
    will be apply to each the images in the batch.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The
            factor is used to determine the lower bound and upper bound of the
            brightness adjustment. A float value will be chosen randomly between
            the limits. When -1.0 is chosen, the output image will be black, and
            when 1.0 is chosen, the image will be fully white.
            When only one float is provided, eg, 0.2,
            then -0.2 will be used for lower bound and 0.2
            will be used for upper bound.
        value_range: Optional list/tuple of 2 floats
            for the lower and upper limit
            of the values of the input data.
            To make no change, use [0.0, 1.0], e.g., if the image input
            has been scaled before this layer. Defaults to [0.0, 255.0].
            The brightness adjustment will be scaled to this range, and the
            output values will be clipped to this range.
        seed: optional integer, for fixed RNG behavior.

    Inputs: 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel
        values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)

    Output: 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the
        `factor`. By default, the layer will output floats.
        The output value will be clipped to the range `[0, 255]`,
        the valid range of RGB colors, and
        rescaled based on the `value_range` if needed.
    """

    _FACTOR_VALIDATION_ERROR = (
        "The `factor` argument should be a number (or a list of two numbers) "
        "in the range [-1.0, 1.0]. "
    )
    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )

    def __init__(self, factor, value_range=(0, 255), seed=7):
        self._set_factor(factor)
        self._set_value_range(value_range)
        self.random_generator = tf.random.Generator.from_seed(seed)

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR + f"Got {value_range}"
            )
        if len(value_range) != 2:
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR + f"Got {value_range}"
            )
        self._value_range = sorted(value_range)

    def _set_factor(self, factor):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR + f"Got {factor}"
                )
            self._check_factor_range(factor[0])
            self._check_factor_range(factor[1])
            self._factor = sorted(factor)
        elif isinstance(factor, (int, float)):
            self._check_factor_range(factor)
            factor = abs(factor)
            self._factor = [-factor, factor]
        else:
            raise ValueError(self._FACTOR_VALIDATION_ERROR + f"Got {factor}")

    def _check_factor_range(self, input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR + f"Got {input_number}"
            )

    def __call__(self, data, train_flag=True):
        if train_flag:
            return self._brightness_adjust(data)
        else:
            return data

    def _brightness_adjust(self, images):
        rank = images.shape.rank
        if rank == 3:
            rgb_delta_shape = (1, 1, 1)
        elif rank == 4:
            # Keep only the batch dim. This will ensure to have same adjustment
            # with in one image, but different across the images.
            rgb_delta_shape = [tf.shape(images)[0], 1, 1, 1]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Got "
                f"inputs.shape = {images.shape}"
            )
        rgb_delta = self.random_generator.uniform(
            shape=rgb_delta_shape,
            minval=self._factor[0],
            maxval=self._factor[1],
        )
        rgb_delta = rgb_delta * (self._value_range[1] - self._value_range[0])
        rgb_delta = tf.cast(rgb_delta, images.dtype)
        images += rgb_delta
        return tf.clip_by_value(
            images, self._value_range[0], self._value_range[1]
        )