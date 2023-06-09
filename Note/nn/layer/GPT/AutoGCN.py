import tensorflow as tf
from Note.nn.initializer import i
from Note.nn.layer.dense import dense
from Note.nn.activation import activation_dict


'''
AutoGCN is a neural network layer that can automatically adjust the kernel size and shape of 
a graph convolution based on the input graph structure and features. 
It applies a structure function to the input graph to compute its structure features, such as degree, 
clustering coefficient, and shortest path length. It then applies a kernel function to map the structure 
features to the kernel size and shape. It also applies an init function to initialize the kernel weight and 
bias, and a gcn function to implement the graph convolution. 
The layer can also apply an activation function to the output features if specified.
'''
class AutoGCN:
    def __init__(self,in_features,out_features,structure_function=None,kernel_function=None,init_function=None,gcn_function=None,activation=None,weight_initializer='Xavier',bias_initializer='zero',dtype='float32',use_bias=True):
        # initialize the auto gcn layer with some parameters 
        # in_features: the size of the input features
        # out_features: the size of the output features
        # structure_function: the function to compute the structure features of the input graph
        # kernel_function: the function to map the structure features to the kernel size and shape
        # init_function: the function to initialize the kernel weight and bias
        # gcn_function: the function to implement the graph convolution
        # activation: the activation function to apply to the output features
        # weight_initializer: the method to initialize the weight matrix for the gcn function
        # bias_initializer: the method to initialize the bias vector for the gcn function
        # dtype: the data type of the tensors
        # use_bias: whether to add a learnable bias to the output
        self.activation=activation_dict[activation]
        self.use_bias=use_bias
        self.weight_list=[]
        if structure_function is None:
            # use a default structure function that computes some basic statistics of the input graph
            self.structure_function = lambda x: tf.stack([
                tf.math.reduce_mean(tf.math.reduce_sum(x, axis=-1)), # average degree
                tf.math.reduce_mean(tf.linalg.trace(tf.matmul(x, x)) / tf.math.reduce_sum(x, axis=-1)), # average clustering coefficient
                tf.math.reduce_mean(tf.math.reduce_min(tf.linalg.expm(x), axis=-1)), # average shortest path length
            ], axis=-1)
        else:
            # use a user-defined structure function
            self.structure_function=structure_function
        if kernel_function is None:
            # use a default kernel function that applies a linear transformation and a ReLU activation
            self.kernel_function=dense([1,2],weight_initializer=weight_initializer,bias_initializer=bias_initializer,activation='relu',dtype=dtype)
        else:
            # use a user-defined kernel function
            self.kernel_function=kernel_function
        self.weight_list.append(self.kernel_function.weight_list)
        if init_function is None:
            # use a default init function that initializes kernel weight and bias randomly
            self.init_function=lambda x:(i.initializer([x.shape[-1],out_features],weight_initializer,dtype),i.initializer([self.out_features],bias_initializer,dtype))
        else:
            # use a user-defined init function
            self.init_function=init_function
        if gcn_function is None:
            # use a default gcn function that implements standard gcn formula with optional bias and activation
            self.gcn_weight=i.initializer([in_features,out_features],weight_initializer,dtype) # initialize the weight matrix for the gcn function with the given initializer and data type
            self.weight_list=[self.gcn_weight] # store the weight matrix in a list for later use
            if use_bias:
                self.gcn_bias=i.initializer([out_features],bias_initializer,dtype) # initialize the bias vector for the gcn function with zeros and the given data type
                self.weight_list.append(self.gcn_bias) # add the bias vector to the weight list
            self.gcn_function=self.gcn
        else:
            # use a user-defined gcn function
            self.gcn_function=gcn_function
            self.weight_list.append(self.gcn_function.weight_list)
    
    
    def gcn(self,x,kernel):
        kernel_size,kernel_shape=kernel # unpack kernel size and shape from kernel vector 
        kernel_size=tf.cast(kernel_size,tf.int32) # cast kernel size to integer type 
        kernel_shape=tf.cast(kernel_shape,tf.int32) # cast kernel shape to integer type 
        x=tf.reshape(x,[x.shape[0],-1,x.shape[-1]]) # reshape input features to [batch_size, num_nodes, in_features]
        x=tf.image.resize(x,[kernel_size,x.shape[-1]],method='nearest') # resize input features to [batch_size, kernel_size, in_features] using nearest neighbor interpolation
        x=tf.reshape(x, [x.shape[0], -1]) # flatten input features to [batch_size, kernel_size * in_features]
        x=tf.matmul(x,self.gcn_weight) # apply a linear transformation to input features using gcn weight matrix 
        if self.use_bias:
            x=x+self.gcn_bias # add a bias vector to output features 
        x=tf.reshape(x,[x.shape[0],kernel_shape,-1]) # reshape output features to [batch_size, kernel_shape, out_features]
        x=tf.math.reduce_mean(x,axis=1) # average output features along kernel shape axis 
        return x # return output features 
    
    
    def output(self,data):
        # define the call function to compute the output features from the input features
        # data: a tensor of shape [batch_size, num_nodes, num_nodes] representing the adjacency matrix of the input graph
        # return: a tensor of shape [batch_size, out_features] representing the output features of the input graph
        structure=self.structure_function(data) # compute the structure features of the input graph by applying structure function to adjacency matrix 
        kernel=self.kernel_function.output(structure) # compute the kernel size and shape by applying kernel function to structure features 
        weight,bias=self.init_function(data) # initialize kernel weight and bias by applying init function to adjacency matrix 
        output=self.gcn_function(data,kernel) # compute output features by applying gcn function to adjacency matrix and kernel vector 
        if self.activation is not None:
            output=self.activation(output) # apply activation function to output features 
        return output
