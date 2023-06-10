import tensorflow as tf
import Note.nn.initializer as i


class maxout:
    def __init__(self,input_dim,output_dim,num_units,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True):
        # input_dim: the dimension of the input features
        # output_dim: the dimension of the output features
        # num_units: the number of linear units per output unit
        # weight_initializer: the initializer for the weight matrix
        # bias_initializer: the initializer for the bias vector
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.num_units=num_units
        # initialize the weight matrix and the bias vector
        self.weight=i.initializer([input_dim,output_dim*num_units],weight_initializer,dtype)
        self.bias=i.initializer([output_dim*num_units],bias_initializer,dtype)
        if use_bias==True:
            self.weight_list=[self.weight,self.bias]
        else:
            self.weight_list=[self.weight]
    
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, input_dim], where N is the number of samples
        # compute the linear transformation and add bias
        linear_output=tf.matmul(data,self.weight)
        linear_output=tf.nn.bias_add(linear_output,self.bias)
        # reshape the linear output to [N, output_dim, num_units]
        linear_output=tf.reshape(linear_output,[-1,self.output_dim,self.num_units])
        # compute the maxout operation along the last dimension
        maxout_output=tf.reduce_max(linear_output,axis=-1)
        return maxout_output
