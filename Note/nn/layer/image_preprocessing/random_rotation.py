import tensorflow as tf
import numpy as np
from Note.nn.layer.image_preprocessing.transform import transform

H_AXIS = -3
W_AXIS = -2

class random_rotation:
    """A preprocessing layer which randomly rotates images during training.

    This layer will apply random rotations to each image, filling empty space
    according to `fill_mode`.

    By default, random rotations are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    rotations at inference time, set `training` to True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        factor: a float represented as fraction of 2 Pi, or a tuple of size 2
            representing lower and upper bound for rotating clockwise and
            counter-clockwise. A positive values means rotating
            counter clock-wise,
            while a negative value means clock-wise.
            When represented as a single
            float, this value is used for both the upper and lower bound.
            For instance, `factor=(-0.2, 0.3)`
            results in an output rotation by a random
            amount in the range `[-20% * 2pi, 30% * 2pi]`.
            `factor=0.2` results in an
            output rotating by a random amount
            in the range `[-20% * 2pi, 20% * 2pi]`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
            - *reflect*: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about
                the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)`
                The input is extended by
                filling all values beyond the edge with
                the same constant value k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
                wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
    """

    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=7,
        fill_value=0.0,
    ):
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = -factor
            self.upper = factor
        if self.upper < self.lower:
            raise ValueError(
                "`factor` argument cannot have a negative value. "
                f"Received: factor={factor}"
            )
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.random_generator = tf.random.Generator.from_seed(seed)

    def __call__(self, data, train_flag=True):
        def random_rotated_inputs(data):
            """Rotated inputs with random ops."""
            original_shape = data.shape
            unbatched = data.shape.rank == 3
            # The transform op only accepts rank 4 inputs,
            # so if we have an unbatched image,
            # we need to temporarily expand dims to a batch.
            if unbatched:
                data = tf.expand_dims(data, 0)
            inputs_shape = tf.shape(data)
            batch_size = inputs_shape[0]
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            min_angle = self.lower * 2.0 * np.pi
            max_angle = self.upper * 2.0 * np.pi
            angles = self.random_generator.uniform(
                shape=[batch_size], minval=min_angle, maxval=max_angle
            )
            output = transform(
                data,
                get_rotation_matrix(angles, img_hd, img_wd),
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                interpolation=self.interpolation,
            )
            if unbatched:
                output = tf.squeeze(output, 0)
            output.set_shape(original_shape)
            return output

        if train_flag:
            return random_rotated_inputs(data)
        else:
            return data

def get_rotation_matrix(angles, image_height, image_width):
    """Returns projective transform(s) for the given angle(s).

    Args:
        angles: A scalar angle to rotate all images by,
            or (for batches of images) a vector with an angle to
            rotate each image in the batch. The rank must be
            statically known (the shape is not `TensorShape(None)`).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.

    Returns:
        A tensor of shape (num_images, 8).
            Projective transforms which can be given
            to operation `image_projective_transform_v2`.
            If one row of transforms is
            [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    x_offset = (
        (image_width - 1)
        - (
            tf.cos(angles) * (image_width - 1)
            - tf.sin(angles) * (image_height - 1)
        )
    ) / 2.0
    y_offset = (
        (image_height - 1)
        - (
            tf.sin(angles) * (image_width - 1)
            + tf.cos(angles) * (image_height - 1)
        )
    ) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.cos(angles)[:, None],
            -tf.sin(angles)[:, None],
            x_offset[:, None],
            tf.sin(angles)[:, None],
            tf.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.float32),
        ],
        axis=1,
    )