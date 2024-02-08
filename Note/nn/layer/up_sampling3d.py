import tensorflow as tf

class up_sampling3d:
    def __init__(self, size):
        # Convert size to a tuple if it is an integer
        if isinstance(size, int):
            size = (size, size, size)
        self.size = size

    def __call__(self, inputs):
        # Repeat each spatial dimension size times along the depth, height and width axes
        outputs = tf.repeat(inputs, self.size[0], axis=1)
        outputs = tf.repeat(outputs, self.size[1], axis=2)
        outputs = tf.repeat(outputs, self.size[2], axis=3)
        return outputs