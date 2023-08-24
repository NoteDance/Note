import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class AdaptiveConv2D:
    def __init__(self,input_shape,filters,kernel_size,strides=(1,1),padding='VALID',activation=None,kernel_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True):
        # filters: the number of output filters
        # kernel_size: the size of the convolution kernel
        # activation: the activation function, default is None
        # use_bias: whether to use bias term, default is True
        # kernel_initializer: the initializer for the kernel weights, default is 'glorot_uniform'
        # bias_initializer: the initializer for the bias term, default is 'zeros'
        self.strides=strides
        self.padding=padding
        if activation is not None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel=i.initializer([kernel_size[0],kernel_size[1],input_shape[-1],filters],kernel_initializer,dtype)
        self.attention=i.initializer([1,1,input_shape[-1],filters],kernel_initializer,dtype)
        self.output_size=filters
        self.param_list=[self.kerne,self.attention]
        if use_bias:
            self.bias=i.initializer([filters],bias_initializer,dtype)
            self.param.append(self.bias)
    
    
    def output(self,data):
        # data: the input tensor, shape [batch_size, height, width, channels]
        # strides: the strides of the convolution
        # padding: the padding mode, either 'VALID' or 'SAME'
        # compute the base convolution output
        conv_output=tf.nn.conv2d(data,self.kernel,strides=self.strides,padding=self.padding)
        if self.use_bias:
            # add the bias term to the convolution output
            conv_output=tf.nn.bias_add(conv_output,self.bias)
        if self.activation is not None:
            # apply the activation function to the convolution output
            conv_output=self.activation(conv_output)
        # compute the attention weights for each spatial location
        att_input=tf.nn.conv2d(data,self.attention,strides=self.strides,padding=self.padding)
        input_shape=tf.shape(att_input) # [batch_size, height, width, filters]
        att_input=tf.reshape(att_input,[-1,tf.math.reduce_prod(input_shape[1:3]),input_shape[-1]])
        att_weights=tf.nn.softmax(att_input,axis=-1)
        att_weights=tf.reshape(att_weights,input_shape)
        # multiply the convolution output with the attention weights element-wise
        adaptive_output=conv_output*att_weights
        return adaptive_output
