import tensorflow as tf


def transform(
    images,
    transforms,
    fill_mode="reflect",
    fill_value=0.0,
    interpolation="bilinear",
    output_shape=None,
):
    """Applies the given transform(s) to the image(s).

    Args:
        images: A tensor of shape
            `(num_images, num_rows, num_columns, num_channels)` (NHWC).
            The rank must be statically known
            (the shape is not `TensorShape(None)`).
        transforms: Projective transform matrix/matrices.
            A vector of length 8 or tensor of size N x 8.
            If one row of transforms is [a0, a1, a2, b0, b1, b2,
            c0, c1], then it maps the *output* point `(x, y)`
            to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
            `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to the
            transform mapping input points to output points.
            Note that gradients are not backpropagated
            into transformation parameters.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        output_shape: Output dimension after the transform, `[height, width]`.
            If `None`, output is the same size as input image.

    Fill mode behavior for each valid value is as follows:

    - `"reflect"`: `(d c b a | a b c d | d c b a)`
    The input is extended by reflecting about the edge of the last pixel.

    - `"constant"`: `(k k k k | a b c d | k k k k)`
    The input is extended by filling all
    values beyond the edge with the same constant value k = 0.

    - `"wrap"`: `(a b c d | a b c d | a b c d)`
    The input is extended by wrapping around to the opposite edge.

    - `"nearest"`: `(a a a a | a b c d | d d d d)`
    The input is extended by the nearest pixel.

    Input shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.

    Output shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.

    Returns:
        Image(s) with the same type and shape as `images`, with the given
        transform(s) applied. Transformed coordinates outside of the input image
        will be filled with zeros.
    """
    if output_shape is None:
        output_shape = tf.shape(images)[1:3]
        if not tf.executing_eagerly():
            output_shape_value = tf.get_static_value(output_shape)
            if output_shape_value is not None:
                output_shape = output_shape_value

    output_shape = tf.convert_to_tensor(
        output_shape, tf.int32)

    if not output_shape.get_shape().is_compatible_with([2]):
        raise ValueError(
            "output_shape must be a 1-D Tensor of 2 elements: "
            "new_height, new_width, instead got "
            f"output_shape={output_shape}"
        )

    fill_value = tf.convert_to_tensor(
        fill_value, tf.float32)

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        output_shape=output_shape,
        fill_value=fill_value,
        transforms=transforms,
        fill_mode=fill_mode.upper(),
        interpolation=interpolation.upper(),
    )