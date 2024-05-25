import tensorflow as tf # import the TensorFlow library
from Note import nn
from Note.nn.Model import Model


class depthwise_conv2d: # define a class for depthwise convolutional layer
    def __init__(self,kernel_size,depth_multiplier=1,input_size=None,strides=[1,1],padding='VALID',weight_initializer='Xavier',bias_initializer='zeros',activation=None,data_format='NHWC',dilations=None,use_bias=True,trainable=True,dtype='float32',): # define the constructor method
        if isinstance(kernel_size,int):
            kernel_size=[kernel_size,kernel_size]
        self.kernel_size=kernel_size
        self.depth_multiplier=depth_multiplier
        self.input_size=input_size
        if isinstance(strides,int):
            self.strides=(1,)+(strides,)*2+(1,)
        else:
            self.strides=(1,) + tuple(strides) + (1,)
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
            self.weight=nn.initializer([kernel_size[0],kernel_size[1],input_size,depth_multiplier],weight_initializer,dtype) # initialize the weight tensor
            if use_bias==True: # if use bias is True
                self.bias=nn.initializer([depth_multiplier*input_size],bias_initializer,dtype) # initialize the bias vector
            if use_bias==True: # if use bias is True
                self.param=[self.weight,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.weight] # store only the weight in a list
            if trainable==False:
                self.param=[]
            Model.param.extend(self.param)
    
    
    def build(self):
        self.weight=nn.initializer([self.kernel_size[0],self.kernel_size[1],self.input_size,self.depth_multiplier],self.weight_initializer,self.dtype) # initialize the weight tensor
        if self.use_bias==True: # if use bias is True
            self.bias=nn.initializer([self.depth_multiplier*self.input_size],self.bias_initializer,self.dtype) # initialize the bias vector
        if self.use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
        if self.trainable==False:
            self.param=[]
        Model.param.extend(self.param)
        return
    
    
    def __call__(self,data): # define the output method
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        if self.use_bias==True: # if use bias is True
            return nn.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d,self.bias) # return the output of applying activation function to the depthwise convolution of data and weight, plus bias
        else: # if use bias is False
            return nn.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d) # return the output of applying activation function to the depthwise convolution of data and weight