import tensorflow as tf

class thresholded_relu:
    """DEPRECATED."""

    def __init__(self, theta=1.0, dtype='float32'):
        if theta is None:
            raise ValueError(
                "Theta of a Thresholded ReLU layer cannot be None, expecting a "
                f"float. Received: {theta}"
            )
        if theta < 0:
            raise ValueError(
                "The theta value of a Thresholded ReLU layer "
                f"should be >=0. Received: {theta}"
            )
        self.theta = tf.convert_to_tensor(theta, dtype=dtype)
        self.dtype = dtype

    def __call__(self, data):
        if data.dtype!=self.dtype:
            data = tf.cast(data, self.dtype)
        return data * tf.greater(data, self.theta)