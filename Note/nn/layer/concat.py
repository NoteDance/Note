import tensorflow as tf


class concat:
    def __init__(self,axis=-1):
        self.axis=axis
        self.save_data_count=None
        
        
    def concat(self,data):
        output=data[0]
        for i in range(1,self.save_data_count):
            output=tf.concat([output,data[i]],axis=self.axis)
        return output
