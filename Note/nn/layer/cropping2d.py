import tensorflow as tf


class cropping2d:
    def __init__(self, cropping=1):
        if isinstance(cropping, int):
            self.cropping = tf.constant([[0, 0], [cropping, cropping], [cropping, cropping], [0, 0]])
        elif isinstance(cropping, list) and len(cropping) == 2:
            self.cropping = tf.constant([[0, 0], [cropping[0], cropping[0]], [cropping[1], cropping[1]], [0, 0]])
        elif isinstance(cropping, list) and len(cropping) == 4:
            self.cropping = tf.constant([[0, 0], [cropping[0], cropping[1]], [cropping[2], cropping[3]], [0, 0]])
        else:
            raise ValueError("Invalid cropping argument. It should be an int or a list of two or four ints.")
    
    
    def __call__(self, data):
        return tf.slice(data, begin=[0, self.cropping[1][0], self.cropping[2][0], 0], size=[-1, -1 - self.cropping[1][0] - self.cropping[1][1], -1 - self.cropping[2][0] - self.cropping[2][1], -1])