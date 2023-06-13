import tensorflow as tf
import Note.nn.activation as a
from Note.nn.initializer import initializer


class group_conv2d:
    def __init__(self,weight_shape,num_groups,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True):
        self.num_groups=num_groups
        self.weight=[]
        self.bias=[]
        self.num_groups=num_groups if weight_shape[-2]%num_groups==0 else 1
        for i in range(num_groups):
            self.weight.append(initializer(weight_shape[:-1]+[weight_shape[-1]//num_groups],weight_initializer,dtype))
            if use_bias==True:
                self.bias.append(initializer([weight_shape[-1]//num_groups],bias_initializer,dtype))
        self.activation=activation
        self.use_bias=use_bias
        if use_bias==True:
            self.param=self.weight+self.bias
        else:
            self.param=self.weight
    
    
    def output(self,data,strides,padding='VALID',data_format='NHWC',dilations=None):
        input_groups=tf.split(data,self.num_groups,axis=-1)
        output_groups=[]
        for i in range(self.num_groups):
            if self.use_bias==True:
                output=a.activation_conv(input_groups[i],self.weight[i],self.activation,strides,padding,data_format,dilations,tf.nn.conv2d,self.bias[i])
            else:
                output=a.activation_conv(input_groups[i],self.weight[i],self.activation,strides,padding,data_format,dilations,tf.nn.conv2d)
            output_groups.append(output)
        output=tf.concat(output_groups,axis=-1)
        return output