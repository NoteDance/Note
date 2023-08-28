import tensorflow as tf


class zeropadding2d:
    def __init__(self, padding=1):
        if isinstance(padding, int):
            self.padding = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])
        elif isinstance(padding, list) and len(padding) == 2:
            self.padding = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
        elif isinstance(padding, list) and len(padding) == 4:
            self.padding = tf.constant([[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]])
        else:
            raise ValueError("Invalid padding argument. It should be an int or a list of two or four ints.")
    
    
    def output(self, data):
        return tf.pad(data, self.padding)