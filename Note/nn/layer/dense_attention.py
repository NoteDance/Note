import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class dense_attention:
    def __init__(self,input_dim,output_dim,weight_initializer='Xavier',bias_initializer='zeros',activation='relu',dtype='float32',use_bias=True):
        # input_dim: the dimension of the input features
        # output_dim: the dimension of the output features
        # weight_initializer: the initializer for the weight matrices
        # bias_initializer: the initializer for the bias vectors
        # activation: the activation function for the attention scores, default is None
        # dtype: the data type of the parameters
        # use_bias: whether to use bias or not
        self.input_dim=input_dim
        self.output_dim=output_dim
        if activation is not None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        # initialize the weight matrices and the bias vectors for the query, key and value projections
        self.weight_Q=i.initializer([input_dim,output_dim],weight_initializer,dtype)
        self.bias_Q=i.initializer([output_dim],bias_initializer,dtype)
        self.weight_K=i.initializer([input_dim,output_dim],weight_initializer,dtype)
        self.bias_K=i.initializer([output_dim],bias_initializer,dtype)
        self.weight_V=i.initializer([input_dim,output_dim],weight_initializer,dtype)
        self.bias_V=i.initializer([output_dim],bias_initializer,dtype)
        if use_bias==True:
            self.param_list=[self.weight_Q,self.bias_Q,self.weight_K,self.bias_K,self.weight_V,self.bias_V]
        else:
            self.param_list=[self.weight_Q,self.weight_K,self.weight_V]
    
    
    def output(self,data):
        # data: a tensor that represents the input features, shape is [N, input_dim], where N is the number of samples
        # compute the query, key and value projections
        query=tf.matmul(data,self.weight_Q)
        query=tf.nn.bias_add(query,self.bias_Q)
        key=tf.matmul(data,self.weight_K)
        key=tf.nn.bias_add(key,self.bias_K)
        value=tf.matmul(data,self.weight_V)
        value=tf.nn.bias_add(value,self.bias_V)
        # compute the attention scores by dot product of query and key, and apply activation function if any
        scores=tf.matmul(query,key,transpose_b=True)
        if self.activation is not None:
            scores=self.activation(scores)
        # normalize the scores by softmax function
        scores=tf.nn.softmax(scores,axis=-1)
        # compute the output by weighted sum of value and scores
        output=tf.matmul(scores,value)
        return output