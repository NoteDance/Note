import tensorflow as tf
import Note.nn.activation as a
import Note.nn.initializer as i


class separable_conv2d:
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True):
        self.weight_D=i.initializer(weight_shape+[1],weight_initializer,dtype)
        self.weight_P=i.initializer([1,1,weight_shape[-1],weight_shape[-1]],weight_initializer,dtype)
        if use_bias==True:
            self.bias_D=i.initializer([weight_shape[-1]],bias_initializer,dtype)
            self.bias_P=i.initializer([weight_shape[-1]],bias_initializer,dtype)
        self.activation=activation
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight_D,self.bias_D,self.weight_P,self.bias_P]
        else:
            self.weight_list=[self.weight_D,self.weight_P]
    
    
    def output(self,data,strides,padding='VALID',data_format='NHWC',dilations=None):
        if self.use_bias==True:
            output=a.activation_conv(data,self.weight_D,self.activation,strides,padding,data_format,dilations,tf.nn.depthwise_conv2d,self.bias_D)
            output=a.activation_conv(output,self.weight_P,self.activation,[1,1,1,1],padding,data_format,None,tf.nn.conv2d,self.bias_P)
            return output
        else:
            output=a.activation_conv(data,self.weight_D,self.activation,strides,padding,data_format,dilations,tf.nn.depthwise_conv2d)
            output=a.activation_conv(output,self.weight_P,self.activation,[1,1,1,1],padding,data_format,None,tf.nn.conv2d)
            return output
