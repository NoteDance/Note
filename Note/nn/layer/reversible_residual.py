import tensorflow as tf


class reversible_residual:
    def __init__(self, f, g):
        self.f=f
        self.g=g
    
    
    def __call__(self, data):
        data1, data2 = tf.split(data, 2, axis=-1) # split the input into two halves
        output1 = data1 + self.f(data2) # compute the first output half
        output2 = data2 + self.g(output1) # compute the second output half
        output = tf.concat([output1, output2], axis=-1) # concatenate the output halves
        return output