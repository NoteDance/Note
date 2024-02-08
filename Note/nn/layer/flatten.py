import tensorflow as tf


class flatten:
    def __init__(self):
        self.output_size=None
    
    
    def __call__(self,data):
        data_shape=tf.shape(data)
        batch_size=data_shape[0]
        num_elements=tf.reduce_prod(data_shape[1:])
        output=tf.reshape(data,[batch_size,num_elements])
        return output