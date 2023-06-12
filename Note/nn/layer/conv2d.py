import tensorflow as tf
import Note.nn.activation as a
import Note.nn.initializer as i


class conv2d:
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True):
        self.weight=i.initializer(weight_shape,weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([weight_shape[-1]],bias_initializer,dtype)
        self.activation=activation
        self.use_bias=use_bias
        if use_bias==True:
            self.param_list=[self.weight,self.bias]
        else:
            self.param_list=[self.weight]
    
    
    def output(self,data,strides,padding='VALID',data_format='NHWC',dilations=None):
        if self.use_bias==True:
            return a.activation_conv(data,self.weight,self.activation,strides,padding,data_format,dilations,tf.nn.conv2d,self.bias)
        else:
            return a.activation_conv(data,self.weight,self.activation,strides,padding,data_format,dilations,tf.nn.conv2d)