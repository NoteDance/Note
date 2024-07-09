import tensorflow as tf # import the TensorFlow library
from Note import nn


class GRUCell: # define a class for gated recurrent unit (GRU) cell
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        self.weight=nn.initializer([weight_shape[0]+weight_shape[1],3*weight_shape[1]],weight_initializer,dtype,trainable) # initialize the weight matrix for all gates and candidate hidden state
        if use_bias==True: # if use bias is True
            self.bias=nn.initializer([3*weight_shape[1]],bias_initializer,dtype,trainable) # initialize the bias vector for all gates and candidate hidden state
        self.use_bias=use_bias # set the use bias flag
        self.output_size=weight_shape[-1]
        if use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight matrix in a list
    
    
    def __call__(self,data,state): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        x=tf.concat([data,state],axis=-1) # concatenate the input data and state along the last dimension
        if self.use_bias==True: # if use bias is True
            z=tf.matmul(x,self.weight)+self.bias # calculate the linear transformation of concatenated data and weight matrix, plus bias vector
        else: # if use bias is False
            z=tf.matmul(x,self.weight) # calculate the linear transformation of concatenated data and weight matrix
        r,z,h=tf.split(z,3,axis=-1) # split the linear transformation into three parts: reset gate, update gate and candidate hidden state
        r=tf.nn.sigmoid(r) # apply activation function to the reset gate
        z=tf.nn.sigmoid(z) # apply activation function to the update gate
        h=tf.nn.tanh(h) # apply activation function to the candidate hidden state
        h_new=z*state+(1-z)*h # calculate the new hidden state value by combining the update gate, previous state and candidate hidden state values
        output=h_new # set the output value as the new hidden state value
        return output,h_new # return the output value and the new hidden state value