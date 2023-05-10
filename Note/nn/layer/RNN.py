import tensorflow as tf


class RNN:
    def __init__(self,weight_shape,timestep,weight_func='uniform',bias_func='zero',activation=None,dtype='float64',return_sequence=False,use_bias=True):
        if type(weight_func)==list:
            if weight_func[0]=='normal':
               self.weight_i=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_s=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
            elif weight_func[0]=='uniform':
                self.weight_i=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_s=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
        else:
            if weight_func=='normal':
                if dtype!=None:
                    self.weight_i=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_s=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                else:
                    self.weight_i=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_s=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
            elif weight_func=='uniform':
                if dtype!=None:
                    self.weight_i=tf.Variable(tf.random.uniform(shape=weight_shape,maxval=0.01,dtype=dtype))
                    self.weight_s=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                else:
                    self.weight_i=tf.Variable(tf.random.uniform(shape=weight_shape,maxval=0.01))
                    self.weight_s=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
            elif weight_func=='zero':
                if dtype!=None:
                    self.weight_i=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_s=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                else:
                    self.weight_i=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_s=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
        if use_bias==True and type(bias_func)==list:
            if bias_func[0]=='normal':
               self.bias=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
            elif bias_func[0]=='uniform':
                self.bias=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
        elif use_bias==True:
            if bias_func=='normal':
                if dtype!=None:
                    self.bias=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
            elif bias_func=='uniform':
                if dtype!=None:
                    self.bias=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
            elif bias_func=='zero':
                if dtype!=None:
                    self.bias=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
        self.output_list=[]
        self.state=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.timestep=timestep
        self.activation=activation
        self.return_sequence=return_sequence
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight_i,self.weight_s,self.bias]
        else:
            self.weight_list=[self.weight_i,self.weight_s]
    
    
    def output(self,data):
        if self.use_bias==True:
            if self.activation=='relu':
                for i in range(self.timestep):
                    output=tf.nn.relu(tf.matmul(data[:][:,i],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    return tf.stack(self.output_list,axis=1)
                else:
                    return self.state
            elif self.activation=='sigmoid':
                for i in range(self.timestep):
                    output=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    return tf.stack(self.output_list,axis=1)
                else:
                    return self.state
            elif self.activation=='tanh':
                for i in range(self.timestep):
                    output=tf.nn.tanh(tf.matmul(data[:][:,i],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    return tf.stack(self.output_list,axis=1)
                else:
                    return self.state
        else:
            if self.activation=='relu':
                for i in range(self.timestep):
                    output=tf.nn.relu(tf.matmul(data[:][:,i],self.weight_i)+tf.matmul(self.state,self.weight_s))
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    return tf.stack(self.output_list,axis=1)
                else:
                    return self.state
            elif self.activation=='sigmoid':
                for i in range(self.timestep):
                    output=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_i)+tf.matmul(self.state,self.weight_s))
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    return tf.stack(self.output_list,axis=1)
                else:
                    return self.state
            elif self.activation=='tanh':
                for i in range(self.timestep):
                    output=tf.nn.tanh(tf.matmul(data[:][:,i],self.weight_i)+tf.matmul(self.state,self.weight_s))
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    return tf.stack(self.output_list,axis=1)
                else:
                    return self.state
