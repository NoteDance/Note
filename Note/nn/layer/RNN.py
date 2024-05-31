import tensorflow as tf # import the TensorFlow library
from Note import nn
from Note.nn.activation import activation_dict # import the activation function dictionary from Note.nn package
from Note.nn.Model import Model


class RNN: # define a class for recurrent neural network (RNN) layer
    def __init__(self,output_size,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,return_sequence=False,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.output_list=[] # initialize an empty list for output sequence
        self.state=tf.zeros(shape=[1,output_size],dtype=dtype) # initialize a zero vector for initial state
        self.activation=activation_dict[activation] # get the activation function from the activation dictionary
        self.return_sequence=return_sequence # set the return sequence flag
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=output_size
        if input_size!=None:
            self.weight_i=nn.initializer([input_size,output_size],weight_initializer,dtype,trainable) # initialize the weight matrix for input data
            self.weight_s=nn.initializer([output_size,output_size],weight_initializer,dtype,trainable) # initialize the weight matrix for previous state
            if use_bias==True: # if use bias is True
                self.bias=nn.initializer([output_size],bias_initializer,dtype,trainable) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=[self.weight_i,self.weight_s,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight_i,self.weight_s] # store only the weight matrices in a list
            Model.param.extend(self.param)
    
    
    def build(self):
        self.weight_i=nn.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype,self.trainable) # initialize the weight matrix for input data
        self.weight_s=nn.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype,self.trainable) # initialize the weight matrix for previous state
        if self.use_bias==True: # if use bias is True
            self.bias=nn.initializer([self.output_size],self.bias_initializer,self.dtype,self.trainable) # initialize the bias vector
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight_i,self.weight_s,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight_i,self.weight_s] # store only the weight matrices in a list
        Model.param.extend(self.param)
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        timestep=data.shape[1] # get the number of timesteps from the input data shape
        if self.use_bias==True: # if use bias is True
            for j in range(timestep): # loop over the timesteps
                output=self.activation(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias) # calculate the output value by applying activation function to the linear transformation of input data and previous state, plus bias vector
                self.output_list.append(output) # append the output value to the output list
                self.state=output # update the state value with the output value
            if self.return_sequence==True: # if return sequence is True
                output=tf.stack(self.output_list,axis=1) # stack the output list along the second dimension to form a tensor of shape [batch_size, timestep, hidden_size]
                self.output_list=[] # clear the output list
                return output # return the output tensor
            else: # if return sequence is False
                self.output_list=[] # clear the output list
                return output # return the last output value
        else: # if use bias is False
            for j in range(timestep): # loop over the timesteps
                output=self.activation(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s)) # calculate the output value by applying activation function to the linear transformation of input data and previous state
                self.output_list.append(output) # append the output value to the output list
                self.state=output # update the state value with the output value
            if self.return_sequence==True: # if return sequence is True
                output=tf.stack(self.output_list,axis=1) # stack the output list along the second dimension to form a tensor of shape [batch_size, timestep, hidden_size]
                self.output_list=[] # clear the output list
                return output # return the output tensor
            else: # if return sequence is False
                self.output_list=[] # clear the output list
                return output # return the last output value