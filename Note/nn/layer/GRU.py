import tensorflow as tf


class GRU:
    def __init__(self,weight_shape,timestep,weight_func='uniform',bias_func='zero',dtype='float64',return_sequence=False,use_bias=True):
        if type(weight_func)==list:
            if weight_func[0]=='normal':
               self.weight_r1=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_r2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_z1=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_z2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_h1=tf.Variable(tf.random.normal(shape=weight_shape,mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
               self.weight_h2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],mean=weight_func[1],stddev=weight_func[2],dtype=dtype))
            elif weight_func[0]=='uniform':
                self.weight_r1=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_r2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_z1=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_z2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_h1=tf.Variable(tf.random.uniform(shape=weight_shape,minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
                self.weight_h2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],minval=weight_func[1],maxval=weight_func[2],dtype=dtype))
        else:
            if weight_func=='normal':
                if dtype!=None:
                    self.weight_r1=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_r2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_z1=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_z2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_h1=tf.Variable(tf.random.normal(shape=weight_shape,dtype=dtype))
                    self.weight_h2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                else:
                    self.weight_r1=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_r2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_z1=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_z2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_h1=tf.Variable(tf.random.normal(shape=weight_shape))
                    self.weight_h2=tf.Variable(tf.random.normal(shape=[weight_shape[1],weight_shape[1]]))
            elif weight_func=='uniform':
                if dtype!=None:
                    self.weight_r1=tf.Variable(tf.random.uniform(shape=weight_shape,maxval=0.01,dtype=dtype))
                    self.weight_r2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                    self.weight_z1=tf.Variable(tf.random.uniform(shape=weight_shape,dtype=dtype))
                    self.weight_z2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                    self.weight_h1=tf.Variable(tf.random.uniform(shape=weight_shape,dtype=dtype))
                    self.weight_h2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01,dtype=dtype))
                else:
                    self.weight_r1=tf.Variable(tf.random.uniform(shape=weight_shape))
                    self.weight_r2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
                    self.weight_z1=tf.Variable(tf.random.uniform(shape=weight_shape))
                    self.weight_z2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
                    self.weight_h1=tf.Variable(tf.random.uniform(shape=weight_shape))
                    self.weight_h2=tf.Variable(tf.random.uniform(shape=[weight_shape[1],weight_shape[1]],maxval=0.01))
            elif weight_func=='zero':
                if dtype!=None:
                    self.weight_r1=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_r2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_z1=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_z2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                    self.weight_h1=tf.Variable(tf.zeros(shape=weight_shape,dtype=dtype))
                    self.weight_h2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]],dtype=dtype))
                else:
                    self.weight_r1=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_r2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_z1=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_z2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
                    self.weight_h1=tf.Variable(tf.zeros(shape=weight_shape))
                    self.weight_h2=tf.Variable(tf.zeros(shape=[weight_shape[1],weight_shape[1]]))
        if use_bias==True and type(bias_func)==list:
            if bias_func[0]=='normal':
               self.bias_r=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
               self.bias_z=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
               self.bias_h=tf.Variable(tf.random.normal(shape=[weight_shape[1]],mean=bias_func[1],stddev=bias_func[2],dtype=dtype))
            elif bias_func[0]=='uniform':
                self.bias_r=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
                self.bias_z=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
                self.bias_h=tf.Variable(tf.random.normal(shape=[weight_shape[1]],minval=bias_func[1],maxval=bias_func[2],dtype=dtype))
        elif use_bias==True:
            if bias_func=='normal':
                if dtype!=None:
                    self.bias_r=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_z=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_h=tf.Variable(tf.random.normal(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias_r=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
                    self.bias_z=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
                    self.bias_h=tf.Variable(tf.random.normal(shape=[weight_shape[1]]))
            elif bias_func=='uniform':
                if dtype!=None:
                    self.bias_r=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_z=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_h=tf.Variable(tf.random.uniform(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias_r=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
                    self.bias_z=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
                    self.bias_h=tf.Variable(tf.random.uniform(shape=[weight_shape[1]]))
            elif bias_func=='zero':
                if dtype!=None:
                    self.bias_r=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_z=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                    self.bias_h=tf.Variable(tf.zeros(shape=[weight_shape[1]],dtype=dtype))
                else:
                    self.bias_r=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
                    self.bias_z=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
                    self.bias_h=tf.Variable(tf.zeros(shape=[weight_shape[1]]))
        self.output_list=[]
        self.state=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.H=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.timestep=timestep
        self.return_sequence=return_sequence
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2,self.bias_r,self.bias_z,self.bias_h]
        else:
            self.weight_list=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2]
    
    
    def output(self,data):
        if self.use_bias==True:
            for i in range(self.timestep):
                R=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_r1)+tf.matmul(self.state,self.weight_r2)+self.bias_r)
                Z=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_z1)+tf.matmul(self.state,self.weight_z2)+self.bias_z)
                H_=tf.nn.tanh(tf.matmul(data[:][:,i],self.weight_h1)+tf.matmul(R*self.H,self.weight_h2)+self.bias_h)
                output=Z*self.H+(1-Z)*H_
                self.output_list.append(output)
                self.H=output
                self.state=output
            if self.return_sequence==True:
                output=tf.stack(self.output_list,axis=1)
                self.output_list=[]
                return output
            else:
                return self.state
        else:
            for i in range(self.timestep):
                R=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_r1)+tf.matmul(self.state,self.weight_r2))
                Z=tf.nn.sigmoid(tf.matmul(data[:][:,i],self.weight_z1)+tf.matmul(self.state,self.weight_z2))
                H_=tf.nn.tanh(tf.matmul(data[:][:,i],self.weight_h1)+tf.matmul(R*self.H,self.weight_h2))
                output=Z*self.H+(1-Z)*H_
                self.output_list.append(output)
                self.H=output
                self.state=output
            if self.return_sequence==True:
                output=tf.stack(self.output_list,axis=1)
                self.output_list=[]
                return output
            else:
                return self.state
