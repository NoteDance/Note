import tensorflow as tf # import the TensorFlow library
import Note.nn.activation as a # import the activation module from Note.nn package
from Note.nn.activation import activation_dict
import Note.nn.initializer as i # import the initializer module from Note.nn package
from Note.nn.Module import Module


class separable_conv1d(Module): # define a class for separable convolutional layer
    def __init__(self,filters,kernel_size,depth_multiplier,input_size=None,strides=[1],padding='VALID',data_format='NHWC',dilations=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,use_bias=True,trainable=True,dtype='float32'): # define the constructor method
        self.kernel_size=kernel_size
        self.depth_multiplier=depth_multiplier
        self.input_size=input_size
        self.strides=strides
        self.padding=padding
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.data_format=data_format
        self.dilations=dilations
        self.activation=activation # set the activation function
        self.use_bias=use_bias # set the use bias flag
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=filters
        if input_size!=None:
            depthwise_filter=[kernel_size,input_size,depth_multiplier]
            self.depthwise_kernel=i.initializer(depthwise_filter,weight_initializer,dtype) # initialize the weight matrix for depthwise convolution
            self.pointwise_kernel=i.initializer([1,depthwise_filter[-1]*depthwise_filter[-2],filters],weight_initializer,dtype) # initialize the weight matrix for pointwise convolution
            if use_bias==True: # if use bias is True
                self.bias=i.initializer([filters],bias_initializer,dtype) # initialize the bias vector
                self.param=[self.depthwise_kernel,self.pointwise_kernel,self.bias] # store the parameters in a list
            else: # if use bias is False
                self.param=[self.depthwise_kernel,self.pointwise_kernel] # store only the weight matrices in a list
            if trainable==False:
                self.param=[]
            Module.param.extend(self.param)
    
    
    def build(self):
        depthwise_filter=[self.kernel_size,self.input_size,self.depth_multiplier]
        self.depthwise_kernel=i.initializer(depthwise_filter,self.weight_initializer,self.dtype) # initialize the weight matrix for depthwise convolution
        self.pointwise_kernel=i.initializer([1,depthwise_filter[-1]*depthwise_filter[-2],self.output_size],self.weight_initializer,self.dtype) # initialize the weight matrix for pointwise convolution
        if self.use_bias==True: # if use bias is True
            self.bias=i.initializer([self.output_size],self.bias_initializer,self.dtype) # initialize the bias vector
            self.param=[self.depthwise_kernel,self.pointwise_kernel,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.depthwise_kernel,self.pointwise_kernel] # store only the weight matrices in a list
        if self.trainable==False:
            self.param=[]
        Module.param.extend(self.param)
        return
    
    
    def output(self,data): # define the output method
        strides = (1,) + tuple(self.strides) * 2 + (1,)
        data = tf.expand_dims(data, 1) # add an extra dimension to input data to make it four-dimensional
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, 0)
        pointwise_kernel = tf.expand_dims(self.pointwise_kernel, 0)
        output=a.activation_conv(data,depthwise_kernel,None,strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d) # calculate the output of applying activation function to the depthwise convolution of input data and weight matrix, plus bias vector
        output=a.activation_conv(output,pointwise_kernel,None,[1,1,1,1],'VALID',self.data_format,None,tf.nn.conv2d) # calculate the output of applying activation function to the pointwise convolution of previous output and weight matrix, plus bias vector
        if self.use_bias==True: # if use bias is True
            output+=self.bias
        output = tf.squeeze(output, [1]) # remove the extra dimension from output data to make it three-dimensional
        if self.activation!=None:
            output=activation_dict[self.activation](output)
        return output # return the final output