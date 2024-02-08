import tensorflow as tf

class up_sampling1d:
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs):
        # Repeat each time step size times along the temporal axis
        outputs = tf.repeat(inputs, self.size, axis=1)
        return outputs