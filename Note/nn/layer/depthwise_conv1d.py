import tensorflow as tf # import the TensorFlow library
from Note import nn
from Note.nn.activation import activation_dict


class depthwise_conv1d: # define a class for depthwise convolutional layer
    def __init__(self,kernel_size,depth_multiplier=1,input_size=None,strides=1,padding='VALID',weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NHWC',dilations=None,use_bias=True,trainable=True,dtype='float32',): # define the constructor method
        strides = (1,) + tuple([strides]) * 2 + (1,)
        self.kernel_size=kernel_size
        self.depth_multiplier=depth_multiplier
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
        if input_size!=None:
            self.output_size=depth_multiplier*input_size
            self.depthwise_kernel=nn.initializer([kernel_size,input_size,depth_multiplier],weight_initializer,dtype,trainable) # initialize the weight tensor
            if use_bias==True: # if use bias is True
                self.bias=nn.initializer([depth_multiplier*input_size],bias_initializer,dtype,trainable) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=[self.depthwise_kernel,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.depthwise_kernel] # store only the weight in a list
    
    
    def build(self):
        self.depthwise_kernel=nn.initializer([self.kernel_size,self.input_size,self.depth_multiplier],self.weight_initializer,self.dtype,self.trainable) # initialize the weight tensor
        if self.use_bias==True: # if use bias is True
            self.bias=nn.initializer([self.depth_multiplier*self.input_size],self.bias_initializer,self.dtype,self.trainable) # initialize the bias vector
        if self.use_bias==True: # if use bias is True
            self.param=[self.depthwise_kernel,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.depthwise_kernel] # store only the weight in a list
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        data = tf.expand_dims(data, 1)
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=0)
        output=nn.activation_conv(data,depthwise_kernel,None,self.strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d) # return the output of applying activation function to the depthwise convolution of data and weight
        if self.use_bias==True: # if use bias is True
            output+=self.bias
        output = tf.squeeze(output, [1]) # remove the extra dimension from output data to make it three-dimensional
        if self.activation!=None:
            output=activation_dict[self.activation](output)
        return output