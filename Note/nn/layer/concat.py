import tensorflow as tf


class concat:
    def __init__(self,axis=-1):
        self.axis=axis
        
        
    def concat(self,data1,data2):
        data=[data1]
        data.extend(data2)
        return tf.concat(data,axis=self.axis)
