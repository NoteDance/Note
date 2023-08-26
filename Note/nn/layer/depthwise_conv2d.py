import tensorflow as tf # import the TensorFlow library
import Note.nn.activation as a # import the activation module from Note.nn package
import Note.nn.initializer as i # import the initializer module from Note.nn package


class depthwise_conv2d: # define a class for depthwise convolutional layer
    def __init__(self,weight_shape,strides=[1,1,1,1],padding='VALID',data_format='NHWC',dilations=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True): # define the constructor method
        self.weight=i.initializer(weight_shape,weight_initializer,dtype) # initialize the weight tensor
        if use_bias==True: # if use bias is True
            self.bias=i.initializer([weight_shape[-1]],bias_initializer,dtype) # initialize the bias vector
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilations=dilations
        self.activation=activation # set the activation function
        self.use_bias=use_bias # set the use bias flag
        self.output_size=weight_shape[-1]*weight_shape[-2]
        if use_bias==True: # if use bias is True
            self.param=[self.weight,self.bias] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight] # store only the weight in a list
    
    
    def output(self,data): # define the output method
        if self.use_bias==True: # if use bias is True
            return a.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d,self.bias) # return the output of applying activation function to the depthwise convolution of data and weight, plus bias
        else: # if use bias is False
            return a.activation_conv(data,self.weight,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d) # return the output of applying activation function to the depthwise convolution of data and weight
