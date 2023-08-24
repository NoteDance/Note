import tensorflow as tf # import the TensorFlow library
import Note.nn.activation as a # import the activation module from Note.nn package
import Note.nn.initializer as i # import the initializer module from Note.nn package


class separable_conv2d: # define a class for separable convolutional layer
    def __init__(self,depthwise_filter,out_channels,strides=[1,1,1,1],padding='VALID',data_format='NHWC',dilations=None,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',use_bias=True): # define the constructor method
        self.weight_D=i.initializer(depthwise_filter,weight_initializer,dtype) # initialize the weight matrix for depthwise convolution
        self.weight_P=i.initializer([1,1,depthwise_filter[-1]*depthwise_filter[-2],out_channels],weight_initializer,dtype) # initialize the weight matrix for pointwise convolution
        if use_bias==True: # if use bias is True
            self.bias_D=i.initializer([depthwise_filter[-1]*depthwise_filter[-2]],bias_initializer,dtype) # initialize the bias vector for depthwise convolution
            self.bias_P=i.initializer([out_channels],bias_initializer,dtype) # initialize the bias vector for pointwise convolution
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilations=dilations
        self.activation=activation # set the activation function
        self.use_bias=use_bias # set the use bias flag
        self.output_size=out_channels
        if use_bias==True: # if use bias is True
            self.param=[self.weight_D,self.bias_D,self.weight_P,self.bias_P] # store the parameters in a list
        else: # if use bias is False
            self.param=[self.weight_D,self.weight_P] # store only the weight matrices in a list
    
    
    def output(self,data): # define the output method
        if self.use_bias==True: # if use bias is True
            output=a.activation_conv(data,self.weight_D,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d,self.bias_D) # calculate the output of applying activation function to the depthwise convolution of input data and weight matrix, plus bias vector
            output=a.activation_conv(output,self.weight_P,self.activation,[1,1,1,1],self.padding,self.data_format,None,tf.nn.conv2d,self.bias_P) # calculate the output of applying activation function to the pointwise convolution of previous output and weight matrix, plus bias vector
            return output # return the final output
        else: # if use bias is False
            output=a.activation_conv(data,self.weight_D,self.activation,self.strides,self.padding,self.data_format,self.dilations,tf.nn.depthwise_conv2d) # calculate the output of applying activation function to the depthwise convolution of input data and weight matrix
            output=a.activation_conv(output,self.weight_P,self.activation,[1,1,1,1],self.padding,self.data_format,None,tf.nn.conv2d) # calculate the output of applying activation function to the pointwise convolution of previous output and weight matrix
            return output # return the final output
