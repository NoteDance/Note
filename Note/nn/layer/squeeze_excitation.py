import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict
from Note.nn.Module import Module


class squeeze_excitation(Module):
    def __init__(self,output_size,ratio,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',activation='relu',use_bias=True,dtype='float32'):
        # input_size: the dimension size of the input features
        # output_size: the dimension size of the output features
        # ratio: the reduction ratio for the squeeze operation
        # weight_initializer: the initializer for the weight matrices
        # bias_initializer: the initializer for the bias vectors
        # activation: the activation function for the squeeze operation, default is relu
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.ratio=ratio
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.activation=activation_dict[activation]
        self.use_bias=use_bias
        self.dtype=dtype
        self.output_size=output_size
        if input_size!=None:
            # initialize the weight matrices and the bias vectors for the squeeze and excitation operations
            self.weight_S=i.initializer([input_size,output_size//ratio],weight_initializer,dtype)
            self.bias_S=i.initializer([output_size//ratio],bias_initializer,dtype)
            self.weight_E=i.initializer([output_size//ratio,output_size],weight_initializer,dtype)
            self.bias_E=i.initializer([output_size],bias_initializer,dtype)
            if use_bias==True:
                self.param=[self.weight_S,self.bias_S,self.weight_E,self.bias_E]
            else:
                self.param=[self.weight_S,self.weight_E]
            Module.param.extend(self.param)
    
    
    def build(self):
        # initialize the weight matrices and the bias vectors for the squeeze and excitation operations
        self.weight_S=i.initializer([self.input_size,self.output_size//self.ratio],self.weight_initializer,self.dtype)
        self.bias_S=i.initializer([self.output_size//self.ratio],self.bias_initializer,self.dtype)
        self.weight_E=i.initializer([self.output_size//self.ratio,self.output_size],self.weight_initializer,self.dtype)
        self.bias_E=i.initializer([self.output_size],self.bias_initializer,self.dtype)
        if self.use_bias==True:
            self.param=[self.weight_S,self.bias_S,self.weight_E,self.bias_E]
        else:
            self.param=[self.weight_S,self.weight_E]
        Module.param.extend(self.param)
        return
        
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, H, W, C], where N is the number of samples, H is the height, W is the width, and C is the channel number
        # fix the axis parameter to [1,2] for 4D data
        axis=[1,2]
        # compute the global average pooling of the input features
        squeeze_output=tf.reduce_mean(data,axis=axis)
        # compute the squeeze operation and apply activation function
        squeeze_output=tf.matmul(squeeze_output,self.weight_S)
        squeeze_output=tf.nn.bias_add(squeeze_output,self.bias_S)
        squeeze_output=self.activation(squeeze_output)
        # compute the excitation operation and apply sigmoid function
        excitation_output=tf.matmul(squeeze_output,self.weight_E)
        excitation_output=tf.nn.bias_add(excitation_output,self.bias_E)
        excitation_output=tf.nn.sigmoid(excitation_output)
        # reshape the excitation output to match the input shape of [N, 1, 1, C]
        excitation_output=tf.reshape(excitation_output,[-1,1,1,self.output_dim])
        # compute the output as a scaled version of the input features
        output=data*excitation_output
        # return the output data
        return output
