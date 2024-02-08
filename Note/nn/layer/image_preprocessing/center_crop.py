import tensorflow as tf


H_AXIS = -3
W_AXIS = -2


class center_crop:
    def __init__(self, height, width, dtype='float32'):
        self.height = height
        self.width = width
        self.compute_dtype=dtype


    def __call__(self, data):
        data = tf.cast(data, self.compute_dtype)
        input_shape = tf.shape(data)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width

        def center_crop():
            h_start = tf.cast(h_diff / 2, tf.int32)
            w_start = tf.cast(w_diff / 2, tf.int32)
            return tf.image.crop_to_bounding_box(
                data, h_start, w_start, self.height, self.width
            )

        def upsize():
            outputs = tf.image.resize(
                data, [self.height, self.width], method=tf.image.ResizeMethod.BICUBIC
            )
            # resize will always output float32, so we need to re-cast.
            return tf.cast(outputs, self.compute_dtype)

        return tf.cond(
            tf.reduce_all((h_diff >= 0, w_diff >= 0)), center_crop, upsize
        )


    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape) 