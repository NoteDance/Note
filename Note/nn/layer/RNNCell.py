import tensorflow as tf # import the TensorFlow library
from Note import nn
from Note.nn.activation import activation_dict # import the activation function dictionary from Note.nn package


class RNNCell: # define a class for recurrent neural network (RNN) cell
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',activation=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        self.weight_i=nn.initializer(weight_shape,weight_initializer,dtype,trainable) # initialize the weight matrix for input data
        self.weight_s=nn.initializer([weight_shape[1],weight_shape[1]],weight_initializer,dtype,trainable) # initialize the weight matrix for previous state
        if use_bias==True: # if use bias is True
            self.bias=nn.initializer([weight_shape[1]],bias_initializer,dtype,trainable) # initialize the bias vector
        self.activation=activation_dict[activation] # get the activation function from the activation dictionary
        self.use_bias=use_bias # set the use bias flag
        self.output_size=weight_shape[-1]
        if use_bias==True: # if use bias is True
            self.param=[self.weight_i,self.weight_s,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight_i,self.weight_s] # store only the weight matrices in a list
    
    
    def __call__(self,data,state): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        output=tf.matmul(data,self.weight_i)+tf.matmul(state,self.weight_s) # calculate the linear transformation of input data and previous state
        if self.use_bias==True: # if use bias is True
            output=output+self.bias # add the bias vector to the linear transformation
        if self.activation is not None: # if activation function is not None
            output=self.activation(output) # apply activation function to the linear transformation
        return output # return the output value