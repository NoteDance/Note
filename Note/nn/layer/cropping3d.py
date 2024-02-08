import tensorflow as tf


class cropping3d:
    def __init__(self, cropping=1):
        if isinstance(cropping, int):
            self.cropping = tf.constant([[0, 0], [cropping, cropping], [cropping, cropping], [cropping, cropping], [0, 0]])
        elif isinstance(cropping, list) and len(cropping) == 3:
            self.cropping = tf.constant([[0, 0], [cropping[0], cropping[0]], [cropping[1], cropping[1]], [cropping[2], cropping[2]], [0, 0]])
        elif isinstance(cropping, list) and len(cropping) == 6:
            self.cropping = tf.constant([[0, 0], [cropping[0], cropping[1]], [cropping[2], cropping[3]], [cropping[4], cropping[5]], [0, 0]])
        else:
            raise ValueError("Invalid cropping argument. It should be an int or a list of three or six ints.")
    
    
    def __call__(self, data):
        shape = tf.shape(data)
        size_1 = shape[1] - self.cropping[1][0] - self.cropping[1][1]
        size_2 = shape[2] - self.cropping[2][0] - self.cropping[2][1]
        size_3 = shape[3] - self.cropping[3][0] - self.cropping[3][1]
        return tf.slice(data, begin=[0, self.cropping[1][0], self.cropping[2][0], self.cropping[3][0], 0], size=[-1, size_1, size_2, size_3, -1])