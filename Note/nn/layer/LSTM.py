import tensorflow as tf


class LSTM:
    def __init__(self,weight_shape,timestep,weight_func='uniform',bias_func='zero',dtype='float64',return_sequence=False,use_bias=True):
        if type(weight_func)==list:
            if weight_func[0]=='normal':
               self.weight_i1=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_i2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_f1=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_f2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_o1=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_o2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_c1=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_c2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
            elif weight_func[0]=='uniform':
                self.weight_i1=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_i2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_f1=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_f2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_o1=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_o2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_c1=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_c2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
        else:
            if weight_func=='normal':
                if dtype!=None:
                    self.weight_i1=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_i2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_f1=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_f2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_o1=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_o2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_c1=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_c2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                else:
                    self.weight_i1=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_i2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_f1=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_f2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_o1=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_o2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_c1=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_c2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
            elif weight_func=='uniform':
                if dtype!=None:
                    self.weight_i1=tf.Variable(tf.random.uniform(shape=weight_shape,maxval=0.01,dtype=dtype))
                    self.weight_i2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                    self.weight_f1=tf.Variable(tf.random.uniform(shape=weight_shape,dtype=dtype))
                    self.weight_f2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                    self.weight_o1=tf.Variable(tf.random.uniform(shape=weight_shape,dtype=dtype))
                    self.weight_o2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                    self.weight_c1=tf.Variable(tf.random.uniform(shape=weight_shape,dtype=dtype))
                    self.weight_c2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                else:
                    self.weight_i1=tf.Variable(tf.random.uniform(shape=weight_shape))
                    self.weight_i2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
                    self.weight_f1=tf.Variable(tf.random.uniform(shape=weight_shape))
                    self.weight_f2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
                    self.weight_o1=tf.Variable(tf.random.uniform(shape=weight_shape))
                    self.weight_o2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
                    self.weight_c1=tf.Variable(tf.random.uniform(shape=weight_shape))
                    self.weight_c2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
            elif weight_func=='zero':
                if dtype!=None:
                    self.weight_i1=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_i2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_f1=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_f2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_o1=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_o2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_c1=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_c2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                else:
                    self.weight_i1=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_i2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_f1=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_f2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_o1=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_o2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_c1=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_c2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
        if use_bias==True and type(bias_func)==list:
            if bias_func[0]=='normal':
               self.bias_i=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
               self.bias_f=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
               self.bias_o=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
               self.bias_c=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
            elif bias_func[0]=='uniform':
                self.bias_i=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
                self.bias_f=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
                self.bias_o=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
                self.bias_c=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
        elif use_bias==True:
            if bias_func=='normal':
                if dtype!=None:
                    self.bias_i=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_f=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_o=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_c=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias_i=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
                    self.bias_f=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
                    self.bias_o=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
                    self.bias_c=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
            elif bias_func=='uniform':
                if dtype!=None:
                    self.bias_i=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_f=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_o=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_c=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias_i=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
                    self.bias_f=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
                    self.bias_o=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
                    self.bias_c=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
            elif bias_func=='zero':
                if dtype!=None:
                    self.bias_i=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_f=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_o=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_c=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias_i=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
                    self.bias_f=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
                    self.bias_o=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
                    self.bias_c=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
        self.output_list=[]
        self.state=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.C=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.timestep=timestep
        self.return_sequence=return_sequence
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight_i1,self.weight_f1,self.weight_o1,self.weight_c1,self.weight_i2,self.weight_f2,self.weight_o2,self.weight_c2,self.bias_i,self.bias_f,self.bias_o,self.bias_c]
        else:
            self.weight_list=[self.weight_i1,self.weight_f1,self.weight_o1,self.weight_c1,self.weight_i2,self.weight_f2,self.weight_o2,self.weight_c2]
    
    
    def output(self,data):
        if self.use_bias==True:
            for i in range(self.timestep):
                I=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_i1)+tf.matmul(self.state,self.weight_i2)+self.bias_i)
                F=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_f1)+tf.matmul(self.state,self.weight_f2)+self.bias_f)
                O=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_o1)+tf.matmul(self.state,self.weight_o2)+self.bias_o)
                C_=tf.nn.tanh(tf.matmul(data[:][:,i],self.weight_c1)+tf.matmul(self.state,self.weight_c2)+self.bias_c)
                C=I*C_+self.C*F
                output=O*tf.nn.tanh(C)
                self.output_list.append(output)
                self.C=C
                self.state=output
            if self.return_sequence==True:
                return tf.stack(self.output_list,axis=1)
            else:
                return self.state
        else:
            for i in range(self.timestep):
                I=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_i1)+tf.matmul(self.state,self.weight_i2))
                F=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_f1)+tf.matmul(self.state,self.weight_f2))
                O=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_o1)+tf.matmul(self.state,self.weight_o2))
                C_=tf.nn.tanh(tf.matmul(data[:][:,i],self.weight_c1)+tf.matmul(self.state,self.weight_c2))
                C=I*C_+self.C_*F
                output=O*tf.nn.tanh(C)
                self.output_list.append(output)
                self.C_=C
                self.state=output
            if self.return_sequence==True:
                return tf.stack(self.output_list,axis=1)
            else:
                return self.state
