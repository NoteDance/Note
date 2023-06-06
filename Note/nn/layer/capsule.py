import tensorflow as tf
from Note.nn.initializer import initializer


class capsule:
    def __init__(self,input_shape,num_capsules,dim_capsules,routings=3,weight_initializer='Xavier',dtype='float32'):
        # initialize the capsule layer with some parameters 
        # input_shape: a tuple of (input_num_capsules, input_dim_capsules) 
        # num_capsules: the number of output capsules 
        # dim_capsules: the dimension of output capsules 
        # routings: the number of routing iterations 
        # weight_initializer: the method to initialize the weight matrix 
        # dtype: the data type of the tensors
        self.input_num_capsules=input_shape[1]
        self.input_dim_capsules=input_shape[2]
        self.num_capsules=num_capsules
        self.dim_capsules=dim_capsules
        self.routings=routings
        self.weight=initializer([self.input_num_capsules,self.num_capsules,
                   self.input_dim_capsules,self.dim_capsules],weight_initializer,dtype)
        self.weight_list=[self.weight]
    
    
    def squash(self,inputs):
        # define the squash function to normalize the output vectors
        # inputs: a tensor of shape [batch_size, num_capsules, dim_capsules]
        # return: a tensor of shape [batch_size, num_capsules, dim_capsules]
        norm=tf.norm(inputs,axis=-1,keepdims=True) # compute the norm of each vector
        norm_squared=norm*norm # compute the squared norm of each vector
        return (norm_squared/(1+norm_squared))*(inputs/norm) # apply the squash formula
    
    
    def output(self,inputs):
        # define the output function to compute the output capsules from the input capsules
        # inputs: a tensor of shape [batch_size, input_num_capsules, input_dim_capsules]
        # return: a tensor of shape [batch_size, num_capsules, dim_capsules]
        inputs_expand=tf.expand_dims(inputs,2) # expand the inputs to match the weight matrix shape
        inputs_tiled=tf.tile(inputs_expand,
                               [1,1,self.num_capsules,1]) # tile the inputs to match the weight matrix shape
        inputs_hat=tf.scan(lambda ac,x:tf.matmul(self.weight,x), # compute the predicted output vectors by multiplying the inputs and the weight matrix
                             elems=inputs_tiled,
                             initializer=tf.zeros([
                                 inputs.shape[0],self.input_num_capsules,
                                 self.num_capsules,self.dim_capsules
                             ]))
        b = tf.zeros([inputs.shape[0],self.input_num_capsules,
                      self.num_capsules]) # initialize the coupling coefficients to zero
        for i in range(self.routings):
            c=tf.nn.softmax(b,axis=2) # compute the softmax of the coupling coefficients
            outputs=self.squash(tf.reduce_sum(tf.multiply(c,inputs_hat),axis=1,
                                           keepdims=True)) # compute the weighted sum of the predicted output vectors and apply the squash function
            if i!=self.routings-1:
                b+=tf.reduce_sum(tf.multiply(outputs,inputs_hat),axis=-1) # update the coupling coefficients by adding the agreement between the outputs and the predictions
        return tf.squeeze(outputs) # remove the redundant dimension and return the output capsules