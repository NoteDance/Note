import tensorflow as tf


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


def is_ragged(tensor):
    """Returns true if `tensor` is a ragged tensor or ragged tensor value."""
    return isinstance(
        tensor, (tf.RaggedTensor, tf.compat.v1.ragged.RaggedTensorValue)
    )
    

class resizing:
    """A preprocessing layer which resizes images.

    This layer resizes an image input to a target height and width. The input
    should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
    format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype.
    By default, the layer will output floats.

    This layer can be called on tf.RaggedTensor batches of input images of
    distinct sizes, and will resize the outputs to dense tensors of uniform
    size.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
            Defaults to `"bilinear"`.
        crop_to_aspect_ratio: If True, resize the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            cropped so as to return the
            largest possible window in the image (of size `(height, width)`)
            that matches the target aspect ratio. By default
            (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
    """

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
    ):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.method = method(interpolation)
        self.crop_to_aspect_ratio = crop_to_aspect_ratio

    def __call__(self, data):
        # tf.image.resize will always output float32
        # and operate more efficiently on float32
        # unless interpolation is nearest, in which case ouput type matches
        # input type.
        if self.interpolation == "nearest":
            input_dtype = data.dtype
        else:
            input_dtype = tf.float32
        inputs = tf.cast(data, dtype=input_dtype)
        size = [self.height, self.width]
        if self.crop_to_aspect_ratio:

            def resize_to_aspect(x):
                if is_ragged(data):
                    x = x.to_tensor()
                return tf.image.resize(x, size=size, method=self.method)

            if is_ragged(data):
                size_as_shape = tf.TensorShape(size)
                shape = size_as_shape + inputs.shape[-1:]
                spec = tf.TensorSpec(shape, input_dtype)
                outputs = tf.map_fn(
                    resize_to_aspect, inputs, fn_output_signature=spec
                )
            else:
                outputs = resize_to_aspect(inputs)
        else:
            outputs = tf.image.resize(data, size=size, method=self.method)
            
        return tf.cast(outputs, data.dtype)