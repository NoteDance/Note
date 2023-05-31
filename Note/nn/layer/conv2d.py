import tensorflow as tf
import Note.nn.activation as a
import Note.nn.initializer as i


class conv2d:
    def __init__(self,data,weight_shape,weight_initializer='normal',activation=None,dtype='float64'):
        self.weight=i.initializer(weight_shape,weight_initializer,dtype)
        self.activation=activation
        self.weight_list=[self.weight]
    
    
    def output(self,data,strides,padding='VALID',data_format='NHWC',dilations=None):
        return a.activation_conv(data,self.weight,self.activation,strides,padding,data_format,dilations,tf.nn.conv2d)
