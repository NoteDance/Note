import tensorflow as tf


class concat:
    def __init__(self,axis=-1):
        self.axis=axis
        
        
    def __call__(self,data):
        output=data.pop(0)
        for i in range(1,len(data)):
            output=tf.concat([output,data.pop(0)],axis=self.axis)
        return output
