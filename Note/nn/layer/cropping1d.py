import tensorflow as tf


class cropping1d:
    def __init__(self, cropping=1):
        if isinstance(cropping, int):
            self.cropping = tf.constant([[0, 0], [cropping, cropping], [0, 0]])
        elif isinstance(cropping, list) and len(cropping) == 2:
            self.cropping = tf.constant([[0, 0], [cropping[0], cropping[1]], [0, 0]])
        else:
            raise ValueError("Invalid cropping argument. It should be an int or a list of two ints.")
    
    
    def __call__(self, data):
        shape = tf.shape(data)
        size = shape[1] - self.cropping[1][0] - self.cropping[1][1]
        return tf.slice(data, begin=[0, self.cropping[1][0], 0], size=[-1, size, -1])