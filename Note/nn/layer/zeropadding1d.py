import tensorflow as tf


class zeropadding1d:
    def __init__(self,input_size=None, padding=1):
        if isinstance(padding, int):
            self.padding = tf.constant([[0, 0], [0, 0], [padding, padding], [0, 0]])
        elif isinstance(padding, list) and len(padding) == 2:
            self.padding = tf.constant([[0, 0], [0, 0], [padding[0], padding[1]], [0, 0]])
        else:
            raise ValueError("Invalid padding argument. It should be an int or a list of two ints.")
        self.input_size=input_size
        if input_size!=None:
            self.output_size=input_size
    
    
    def build(self):
        self.output_size=self.input_size
    
    
    def output(self, data):
        return tf.pad(data, self.padding)