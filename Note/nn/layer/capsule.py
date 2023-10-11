import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.Module import Module


class capsule:
    def __init__(self,num_capsules,dim_capsules,input_size=None,routings=3,weight_initializer='Xavier',trainable=True,dtype='float32'):
        # initialize the capsule layer with some parameters
        self.num_capsules=num_capsules # the number of output capsules
        self.dim_capsules=dim_capsules # the dimension of each output capsule
        self.input_size=input_size
        self.routings=routings # the number of routing iterations
        self.weight_initializer=weight_initializer
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=dim_capsules
        if input_size!=None:
            self.weight=initializer([input_size,num_capsules*dim_capsules],weight_initializer,dtype) # the weight matrix for transforming input capsules to output capsules
            self.param=[self.weight] # a list to store the weight matrix
            if trainable==False:
                self.param=[]
            Module.param.extend(self.param)
    
    
    def build(self):
        self.weight=initializer([self.input_size,self.num_capsules*self.dim_capsules],self.weight_initializer,self.dtype) # the weight matrix for transforming input capsules to output capsules
        self.param=[self.weight] # a list to store the weight matrix
        if self.trainable==False:
            self.param=[]
        Module.param.extend(self.param)
        return
    
    
    def squash(self,data):
        # define the squash function to normalize the output vectors
        norm=tf.norm(data,axis=-1,keepdims=True) # compute the norm of each vector
        norm_squared=norm*norm # compute the squared norm of each vector
        return (norm_squared/(1+norm_squared))*(data/norm) # apply the squash formula
    
    
    def output(self,data):
        # define the output function to compute the output capsules from the input data
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        # check the dimension of the input data
        if len(data.shape)==4: # four-dimensional data
            # reshape the data to [batch_size, height * width * channels, input_dim_capsules]
            data=tf.reshape(data,[data.shape[0],-1,data.shape[-1]])
            # update the input_num_capsules accordingly
            self.input_num_capsules=data.shape[1]
        elif len(data.shape)==3: # three-dimensional data
            # no need to reshape the data
            pass
        elif len(data.shape)==2: # two-dimensional data
            # reshape the data to [batch_size, 1, input_dim_capsules]
            data=tf.expand_dims(data,axis=1)
            # update the input_num_capsules accordingly
            self.input_num_capsules=1
        else:
            raise ValueError("Unsupported input shape:{}".format(data.shape))
        data_reshape=tf.reshape(data,[self.batch_size*self.input_num_capsules,self.input_dim_capsules]) # reshape the data to [batch_size * input_num_capsules, input_dim_capsules]
        data_hat=tf.matmul(data_reshape,self.weight) # multiply the data by the weight matrix to get the predictions for output capsules
        data_hat=tf.reshape(data_hat,[self.batch_size,self.input_num_capsules,self.num_capsules,self.dim_capsules]) # reshape the predictions to [batch_size, input_num_capsules, num_capsules, dim_capsules]
        b = tf.zeros([self.batch_size,self.input_num_capsules,
                      self.num_capsules]) # initialize a zero matrix for storing coupling coefficients
        for i in range(self.routings): # iterate for a number of routings
            c=tf.nn.softmax(b,axis=2) # apply softmax on b to get c as a probability distribution over output capsules for each input capsule
            c = tf.expand_dims(c,-1) # expand c along the last dimension to match data_hat shape
            outputs=self.squash(tf.reduce_sum(tf.multiply(c,data_hat),axis=1,
                                                       keepdims=True)) # compute the weighted sum of predictions and apply squash function to get outputs as vectors with length between 0 and 1
            if i!=self.routings-1: # if not the last iteration
                b+=tf.reduce_sum(tf.multiply(data_hat,outputs),axis=-1) # update b by adding the scalar product of outputs and predictions
        return tf.squeeze(outputs) # return outputs after squeezing out redundant dimensions
