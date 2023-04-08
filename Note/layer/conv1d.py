import tensorflow as tf


class conv1d:
    def __init__(self,data,weight_shape,weight_func='normal',activation=None,dtype='float64'):
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
        self.activation=activation
    
    
    def output(self,data,strides,padding='VALID',data_format='NWC',dilations=None):
        if self.activation=='relu':
            return tf.nn.relu(tf.nn.conv2d(data,self.weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif self.activation=='sigmoid':
            return tf.nn.sigmoid(tf.nn.conv2d(data,self.weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif self.activation=='tanh':
            return tf.nn.tanh(tf.nn.conv2d(data,self.weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        else:
            return tf.nn.conv2d(data,self.weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)