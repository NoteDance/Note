import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class highway:
    def __init__(self,input_dim,output_dim,weight_initializer='Xavier',bias_initializer='zeros',activation='relu',dtype='float32',use_bias=True):
        # input_dim: the dimension of the input features
        # output_dim: the dimension of the output features
        # weight_initializer: the initializer for the weight matrices
        # bias_initializer: the initializer for the bias vectors
        # activation: the activation function for the transform gate, default is None
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.input_dim=input_dim
        self.output_dim=output_dim
        if activation is not None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        # initialize the weight matrices and the bias vectors for the transform and carry gates
        self.weight_T=i.initializer([input_dim,output_dim],weight_initializer,dtype)
        self.bias_T=i.initializer([output_dim],bias_initializer,dtype)
        self.weight_C=i.initializer([input_dim,output_dim],weight_initializer,dtype)
        self.bias_C=i.initializer([output_dim],bias_initializer,dtype)
        self.output_size=output_dim
        if use_bias==True:
            self.param=[self.weight_T,self.bias_T,self.weight_C,self.bias_C]
        else:
            self.param=[self.weight_T,self.weight_C]
    
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, input_dim], where N is the number of samples
        # compute the transform gate output and apply activation function
        transform_output=tf.matmul(data,self.weight_T)
        transform_output=tf.nn.bias_add(transform_output,self.bias_T)
        if self.activation is not None:
            transform_output=self.activation(transform_output)
        # compute the carry gate output and apply sigmoid function
        carry_output=tf.matmul(data,self.weight_C)
        carry_output=tf.nn.bias_add(carry_output,self.bias_C)
        carry_output=tf.nn.sigmoid(carry_output)
        # compute the highway output as a linear combination of transform and carry outputs
        highway_output=transform_output*carry_output+data*(1-carry_output)
        return highway_output
