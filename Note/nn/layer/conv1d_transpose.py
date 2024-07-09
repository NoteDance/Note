import tensorflow as tf # import the TensorFlow library
from Note import nn


class conv1d_transpose: # define a class for 1D transposed convolutional layer
    def __init__(self,filters,kernel_size,input_size=None,strides=[1],padding='VALID',output_padding=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NWC',dilations=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        if isinstance(kernel_size,int):
            kernel_size=kernel_size
        elif len(kernel_size)==1:
            kernel_size=kernel_size[0]
        if isinstance(strides,int):
            strides=[strides]
        self.kernel_size=kernel_size
        self.input_size=input_size
        self.strides=strides
        self.padding=padding
        self.output_padding=output_padding
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
            self.weight=nn.initializer([kernel_size,filters,input_size],weight_initializer,dtype,trainable) # initialize the weight tensor
            if use_bias==True: # if use bias is True
                self.bias=nn.initializer([filters],bias_initializer,dtype,trainable) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
            if trainable==False:
                self.param=[]
    
    
    def build(self):
        self.weight=nn.initializer([self.kernel_size,self.output_size,self.input_size],self.weight_initializer,self.dtype,self.trainable) # initialize the weight tensor
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
        timesteps = data.shape[1] # get the number of timesteps in the input data
        if self.padding == 'SAME': # if padding is 'SAME'
            padding = tf.math.ceil((self.kernel_size - 1) / 2) # calculate the padding value as (kernel_size - 1) / 2, rounded up
        else: # if padding is not 'same'
            padding = 0 # set the padding value to 0
        
        if self.output_padding == None: # if output_padding is None
            output_padding = 0 # set the output_padding value to 0
        else: # if output_padding is not None
            output_padding = self.output_padding # use the given output_padding value
        
        new_steps = ((timesteps - 1) * self.strides[0] + self.kernel_size - 2 * padding + output_padding) # calculate the new number of timesteps for the output using the formula
        
        if self.use_bias==True: # if use bias is True
            return nn.activation_conv_transpose(data,self.weight,[data.shape[0],new_steps,self.output_size],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv1d_transpose,bias=self.bias) # return the output of applying activation function to the transposed convolution of data and weight, plus bias
        else: # if use bias is False
            return nn.activation_conv_transpose(data,self.weight,[data.shape[0],new_steps,self.output_size],self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.conv1d_transpose) # return the output of applying activation function to the transposed convolution of data and weight