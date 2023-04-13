import tensorflow as tf


class dense:
    def __init__(self,weight_shape,weight_func='uniform',bias_func='zero',activation=None,dtype='float64',use_bias=True):
        if type(weight_func)==list:
            if weight_func[0]=='normal':
               self.weight=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
            elif weight_func[0]=='uniform':
                self.weight=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
        else:
            if weight_func=='normal':
                if dtype!=None:
                    self.weight=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                else:
                    self.weight=tf.Variable(tf.random.normal(shape=weight_shape))
            elif weight_func=='uniform':
                if dtype!=None:
                    self.weight=tf.Variable(tf.random.uniform(shape=weight_shape,dtype=dtype))
                else:
                    self.weight=tf.Variable(tf.random.uniform(shape=weight_shape))
            elif weight_func=='zero':
                if dtype!=None:
                    self.weight=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                else:
                    self.weight=tf.Variable(tf.zeros(shape=weight_shape))
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
        self.activation=activation
        self.use_bias=use_bias
    
    
    def output(self,data):
        if self.use_bias==True:
            if self.activation=='relu':
                return tf.nn.relu(tf.matmul(data,self.weight)+self.bias)
            elif self.activation=='sigmoid':
                return tf.nn.sigmoid(tf.matmul(data,self.weight)+self.bias)
            elif self.activation=='tanh':
                return tf.nn.tanh(tf.matmul(data,self.weight)+self.bias)
            else:
                return tf.matmul(data,self.weight)+self.bias
        else:
            if self.activation=='relu':
                return tf.nn.relu(tf.matmul(data,self.weight))
            elif self.activation=='sigmoid':
                return tf.nn.sigmoid(tf.matmul(data,self.weight))
            elif self.activation=='tanh':
                return tf.nn.tanh(tf.matmul(data,self.weight))
            else:
                return tf.matmul(data,self.weight)
