import tensorflow as tf
from Note.nn.initializer import i
from Note.nn.activation import activation_dict


'''
The purpose of this layer module is to let the neural network use tree structures 
to process hierarchical data in convolution operations. Tree structures can help the 
neural network capture the parent-child relationships and the global structure of the data. 
This layer module can be used in the following fields:

Natural language processing, such as parsing sentences or generating sentences
Graph neural networks, such as learning graph representations or graph classification
Computer vision, such as segmenting images or detecting objects
Bioinformatics, such as analyzing protein structures or phylogenetic trees
'''
class TreeConv:
    def __init__(self,in_features,out_features,activation=None,weight_initializer='Xavier',bias_initializer='zero',dtype='float32',use_bias=True):
        # initialize the tree convolution layer with some parameters 
        # in_features: the size of the input features
        # out_features: the size of the output features
        # activation: the activation function to apply to the updated node features
        # weight_initializer: the method to initialize the weight matrix
        # dtype: the data type of the tensors
        # use_bias: whether to add a learnable bias to the output
        self.activation=activation_dict[activation]
        self.dtype=dtype
        self.use_bias=use_bias
        self.weight=i.initializer([in_features,out_features],weight_initializer,dtype) # initialize the weight matrix with the given initializer and data type
        self.weight_list=[self.weight] # store the weight matrix in a list for later use
        if use_bias:
            self.bias=i.initializer([out_features],bias_initializer,dtype) # initialize the bias vector with zeros and the given data type
            self.weight_list.append(self.bias) # add the bias vector to the weight list
    
    
    def output(self,data,parent):
        # define the output function to compute the output features from the input features and the parent indices of each node
        # data: a tensor of shape [batch_size, num_nodes, in_features]
        # parent: a tensor of shape [batch_size, num_nodes] indicating the parent index of each node (or -1 for root nodes)
        # return: a tensor of shape [batch_size, num_nodes, out_features]
        output=tf.matmul(data,self.weight) # apply the linear transformation to the input features using the weight matrix
        parent_mask=tf.expand_dims(tf.cast(parent!=-1,self.dtype)-1) # create a mask to indicate which nodes have parents
        parent_data=tf.gather(output,parent,batch_dims=1) # gather the parent features for each node
        output=output+tf.multiply(parent_data,parent_mask) # add the parent features to the node features if they have parents
        if self.use_bias:
            output=output+self.bias # add the bias vector to the output features
        if self.activation is not None:
            output=self.activation(output) # apply the activation function to the output features
        return output