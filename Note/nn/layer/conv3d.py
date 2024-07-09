import tensorflow as tf # import the TensorFlow library
from Note import nn


class conv3d: # define a class for 3D convolutional layer
    def __init__(self,filters,kernel_size,input_size=None,strides=[1,1,1],padding='VALID',weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NDHWC',dilations=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        if isinstance(kernel_size,int):
            kernel_size=[kernel_size,kernel_size,kernel_size]
        if isinstance(strides,int):
            strides=(1,) + tuple((strides,))*3 + (1,)
        else:
            strides=(1,) + tuple(strides) + (1,)
        self.kernel_size=kernel_size
        self.input_size=input_size
        self.strides=strides
        self.padding=padding
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.activation=activation # set the activation function
        self.data_format=data_format
        self.dilations=dilations
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=filters
        if input_size!=None:
            self.weight=nn.initializer([kernel_size[0],kernel_size[1],kernel_size[2],input_size,filters],weight_initializer,dtype,trainable) # initialize the weight tensor
            if use_bias==True: # if use bias is True
                self.bias=nn.initializer([filters],bias_initializer,dtype,trainable) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
            
    
    
    def build(self):
        self.weight=nn.initializer([self.kernel_size[0],self.kernel_size[1],self.kernel_size[2],self.input_size,self.output_size],self.weight_initializer,self.dtype,self.trainable) # initialize the weight tensor
        if self.use_bias==True: # if use bias is True
            self.bias=nn.initializer([self.output_size],self.bias_initializer,self.dtype,self.trainable) # initialize the bias vector
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        if self.use_bias==True: # if use bias is True
            return nn.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv3d,self.bias) # return the output of applying activation function to the convolution of data and weight, plus bias
        else: # if use bias is False
            return nn.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv3d) # return the output of applying activation function to the convolution of data and weight