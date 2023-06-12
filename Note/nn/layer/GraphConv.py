import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class GraphConv:
    def __init__(self,in_features,out_features,norm='both',activation=None,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',use_bias=True):
        # initialize the graph convolution layer with some parameters 
        # in_features: the size of the input features
        # out_features: the size of the output features
        # norm: how to apply the normalizer ('right', 'none', 'both', or 'left')
        # activation: the activation function to apply to the updated node features
        # weight_initializer: the method to initialize the weight matrix
        # dtype: the data type of the tensors
        # use_bias: whether to add a learnable bias to the output
        self.norm=norm
        self.activation=activation_dict[activation]
        self.use_bias=use_bias
        self.weight=i.initializer([in_features,out_features],weight_initializer,dtype) # initialize the weight matrix with the given initializer and data type
        self.weight_list=[self.weight] # store the weight matrix in a list for later use
        if use_bias:
            self.bias=i.initializer([out_features],bias_initializer,dtype) # initialize the bias vector with zeros and the given data type
            self.param_list.append(self.bias) # add the bias vector to the param list
    
    
    def output(self,data,adj):
        # define the output function to compute the output features from the input features and the adjacency matrix
        # data: a tensor of shape [batch_size, num_nodes, in_features]
        # adj: a tensor of shape [batch_size, num_nodes, num_nodes]
        # return: a tensor of shape [batch_size, num_nodes, out_features]
        output=tf.matmul(data,self.weight) # apply the linear transformation to the input features using the weight matrix
        if self.norm=='right':
            adj=tf.divide(adj,tf.reduce_sum(adj,axis=-1,keepdims=True)) # normalize the adjacency matrix by dividing each row by its sum (right normalization)
        elif self.norm=='left':
            adj=tf.divide(adj,tf.reduce_sum(adj,axis=-2,keepdims=True)) # normalize the adjacency matrix by dividing each column by its sum (left normalization)
        elif self.norm=='both':
            deg=tf.pow(tf.reduce_sum(adj,axis=-1),-0.5) # compute the degree vector of each node as the inverse square root of its degree
            deg=tf.expand_dims(deg,-1) # expand the degree vector to match the adjacency matrix shape
            adj=tf.multiply(tf.multiply(deg,adj),deg) # normalize the adjacency matrix by multiplying each element by the product of the square root of node degrees (both normalization)
        output=tf.matmul(adj,output) # apply the graph convolution to the input features using the normalized adjacency matrix
        if self.use_bias:
            output=output+self.bias # add the bias vector to the output features
        if self.activation is not None:
            output=self.activation(output) # apply the activation function to the output features
        return output