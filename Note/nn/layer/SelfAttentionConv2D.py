import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class SelfAttentionConv2D:
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
        self.kernel=i.initializer([kernel_size[0],kernel_size[1],input_shape[-1],filters],kernel_initializer,dtype)
        self.output_size=filters
        self.param=[self.kernel]
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
        # compute the query, key and value tensors for each spatial location
        query=tf.nn.conv2d(data,self.kernel,strides=self.strides,padding=self.padding)
        key=tf.nn.conv2d(data,self.kernel,strides=self.strides,padding=self.padding)
        value=tf.nn.conv2d(data,self.kernel,strides=self.strides,padding=self.padding)
        input_shape=tf.shape(query) # [batch_size, height, width, filters]
        query=tf.reshape(query,[-1,input_shape[1]*input_shape[2],input_shape[-1]])
        key=tf.reshape(key,[-1,input_shape[1]*input_shape[2],input_shape[-1]])
        value=tf.reshape(value,[-1,input_shape[1]*input_shape[2],input_shape[-1]])
        # compute the attention scores for each pair of spatial locations
        att_scores=tf.matmul(query,key,transpose_b=True) # [batch_size, height*width, height*width]
        att_scores=tf.nn.softmax(att_scores,axis=-1) # normalize along the last dimension
        # compute the weighted sum of value tensors for each spatial location
        att_output=tf.matmul(att_scores,value) # [batch_size, height*width, filters]
        att_output=tf.reshape(att_output,input_shape) # [batch_size, height, width, filters]
        # add the attention output to the convolution output element-wise
        self_attention_output=conv_output+att_output
        return self_attention_output
