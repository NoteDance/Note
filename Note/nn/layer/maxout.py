import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.Module import Module


class maxout(Module):
    def __init__(self,output_size,num_units,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',use_bias=True,dtype='float32'):
        # input_size: the dimension size of the input features
        # output_size: the dimension size of the output features
        # num_units: the number of linear units per output unit
        # weight_initializer: the initializer for the weight matrix
        # bias_initializer: the initializer for the bias vector
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.num_units=num_units
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.use_bias=use_bias
        self.dtype=dtype
        self.output_size=output_size
        if input_size!=None:
            # initialize the weight matrix and the bias vector
            self.weight=i.initializer([input_size,output_size*num_units],weight_initializer,dtype)
            self.bias=i.initializer([output_size*num_units],bias_initializer,dtype)
            if use_bias==True:
                self.param=[self.weight,self.bias]
            else:
                self.param=[self.weight]
            Module.param.extend(self.param)
    
    
    def build(self):
        # initialize the weight matrix and the bias vector
        self.weight=i.initializer([self.input_size,self.output_size*self.num_units],self.weight_initializer,self.dtype)
        self.bias=i.initializer([self.output_size*self.num_units],self.bias_initializer,self.dtype)
        if self.use_bias==True:
            self.param=[self.weight,self.bias]
        else:
            self.param=[self.weight]
        Module.param.extend(self.param)
        return
    
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, input_dim], where N is the number of samples
        # compute the linear transformation and add bias
        linear_output=tf.matmul(data,self.weight)
        linear_output=tf.nn.bias_add(linear_output,self.bias)
        # reshape the linear output to [N, output_dim, num_units]
        linear_output=tf.reshape(linear_output,[-1,self.output_dim,self.num_units])
        # compute the maxout operation along the last dimension
        maxout_output=tf.reduce_max(linear_output,axis=-1)
        return maxout_output