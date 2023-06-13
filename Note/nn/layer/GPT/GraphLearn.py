import tensorflow as tf
from Note.nn.initializer import i
from Note.nn.activation import activation_dict


'''
GraphLearn is a neural network layer that can learn the adjacency matrix of a graph 
from its node features. It applies a graph function to the input features and outputs a 
softmax-normalized matrix that represents the learned graph structure. 
The graph function can be customized by the user or use a default linear transformation. 
The layer can also apply an activation function to the output features if specified.
'''
class GraphLearn:
    def __init__(self,in_features,out_features,similarity_function=None,weight_function=None,aggregate_function=None,activation=None,threshold=0.5,norm='both',weight_initializer='Xavier',bias_initializer='zeros',similarity_initializer='Xavier',dtype='float32',use_bias=True):
        # initialize the graph learn layer with some parameters 
        # in_features: the size of the input features
        # out_features: the size of the output features
        # similarity_function: the function to compute the similarity between two nodes
        # weight_function: the function to compute the weight of an edge between two nodes
        # aggregate_function: the function to aggregate the features of neighboring nodes
        # activation: the activation function to apply to the updated node features
        # threshold: the threshold to decide whether two nodes have an edge or not
        # norm: how to apply the normalizer ('right', 'none', 'both', or 'left')
        # weight_initializer: the method to initialize the weight matrix for the aggregation function
        # bias_initializer: the method to initialize the bias vector for the aggregation function
        # similarity_initializer: the method to initialize the weight matrix for the similarity function
        # dtype: the data type of the tensors
        # use_bias: whether to add a learnable bias to the output
        self.threshold=tf.Variable(tf.constant(threshold),trainable=False) # initialize the threshold as a non-trainable variable
        self.norm=norm
        self.activation=activation_dict[activation]
        self.dtype=dtype
        self.use_bias=use_bias
        self.param=[]
        if similarity_function is None:
            # use a default similarity function based on dot product and linear transformation
            self.similarity=i.initializer([in_features,in_features],similarity_initializer,dtype) # initialize the weight matrix for the similarity function with the given initializer and data type
            self.similarity_function=lambda x1,x2:tf.matmul(tf.matmul(x1,self.similarity),tf.transpose(x2,perm=[0,2,1])) 
            self.param.append(self.similarity)
        else:
            # use a user-defined similarity function
            self.similarity_function=similarity_function
            self.param.append(self.similarity_function.weight_list)
        if weight_function is None:
            # use a default weight function based on dot product and linear transformation
            self.weight=i.initializer([in_features,1],weight_initializer,dtype) # initialize a column of weight matrix for the weight function with the given initializer and data type
            self.weight_function=lambda x1,x2:tf.matmul(tf.matmul(x1,tf.expand_dims(self.weight[:,0],-1)),tf.transpose(tf.matmul(x2,tf.expand_dims(self.weight[:,0],-1)),perm=[0,2,1]))
            self.param.append(self.weight)
        else:
            # use a user-defined weight function
            self.weight_function=weight_function
            self.param.append(self.weight_function.weight_list)
        if aggregate_function is None:
            # use a default aggregate function based on linear transformation and optional bias and activation
            self.aggregate_weight=i.initializer([in_features, out_features], weight_initializer, dtype) # initialize the weight matrix for the aggregation function with the given initializer and data type
            self.param.append(self.aggregate_weight) # store the weight matrix in a list for later use
            if use_bias:
                self.aggregate_bias=i.initializer([out_features],bias_initializer,dtype) # initialize the bias vector for the aggregation function with zeros and the given data type
                self.param.append(self.aggregate_bias) # add the bias vector to the weight list
            self.aggregate_function=self.aggregate
        else:
            # use a user-defined aggregate function 
            self.aggregate_function=aggregate_function
            self.param.append(self.aggregate_function.weight_list)
    
    
    def aggregate(self,x):
        output=tf.matmul(x,self.aggregate_weight) # apply a linear transformation to x using the aggregate weight matrix 
        if self.use_bias:
            output=output+self.aggregate_bias # add a bias vector to output 
        if self.activation is not None:
            output=self.activation(output) # apply an activation function to output 
        return output
    
    
    def output(self,data):
        # define the output function to compute the output features from the input features
        # data: a tensor of shape [batch_size, num_nodes, in_features]
        # return: a tensor of shape [batch_size, num_nodes, out_features]
        similarity=self.similarity_function(data,data) # compute the similarity matrix between each pair of nodes using the similarity function
        adjacency=tf.cast(similarity>self.threshold,self.dtype) # compute the adjacency matrix by applying a threshold to the similarity matrix and casting it to the same data type as input features
        weight=self.weight_function(data,data) # compute a weight matrix between each pair of nodes using the weight function
        if self.norm=='right':
            adjacency=tf.divide(adjacency,tf.reduce_sum(adjacency,axis=-1,keepdims=True)+1e-10) # normalize the adjacency matrix by dividing each row by its sum (right normalization) and add a small constant to avoid division by zero
        elif self.norm=='left':
            adjacency=tf.divide(adjacency,tf.reduce_sum(adjacency,axis=-2,keepdims=True)+1e-10) # normalize the adjacency matrix by dividing each column by its sum (left normalization) and add a small constant to avoid division by zero
        elif self.norm=='both':
            degree=tf.pow(tf.reduce_sum(adjacency,axis=-1,keepdims=True)+1e-10,-0.5) # compute a degree vector for each node as the inverse square root of its degree and add a small constant to avoid division by zero
            degree=tf.matmul(degree,tf.transpose(degree,perm=[0,2,1])) # compute a degree matrix between each pair of nodes using their inverse square root degree vectors and dot product 
            adjacency=degree*adjacency # normalize the adjacency matrix by multiplying each element by the product of the square root of node degrees (both normalization)
        laplacian=adjacency*weight # compute a laplacian matrix between each pair of nodes using their adjacency matrix and weight matrix 
        aggregate=tf.matmul(laplacian,data) # apply a graph convolution to the input features using the laplacian matrix 
        output=self.aggregate_function(aggregate) # apply the aggregate function to the aggregated features 
        return output