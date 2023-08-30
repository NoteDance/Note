import tensorflow as tf


class concat:
    def __init__(self,axis=-1):
        self.axis=axis
        
        
    def concat(self,data1,data2):
        return tf.concat([data1,data2],axis=self.axis)
