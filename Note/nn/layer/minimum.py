import tensorflow as tf


class minimum:
    def __init__(self):
        self.save_data_count=None
        
        
    def __call__(self,data):
        if self.save_data_count!=None:
            output=data.pop(0)
            for i in range(1,self.save_data_count):
                output=tf.minimum(output,data.pop(0))
        else:
            output=data[0]
            for i in range(1,len(data)):
                output=tf.minimum(output,data[i])
        return output