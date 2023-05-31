import tensorflow as tf
import Note.nn.initializer as i


class LSTM:
    def __init__(self,weight_shape,timestep,weight_initializer='uniform',bias_initializer='zero',dtype='float64',return_sequence=False,use_bias=True):
        self.weight_i1=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_i2=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_f1=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_f2=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_o1=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_o2=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_c1=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_c2=i.initializer(weight_shape,weight_initializer,dtype)
        if use_bias==True:
            self.bias_i=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_f=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_o=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_c=i.initializer([weight_shape[1]],bias_initializer,dtype)
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
            for j in range(self.timestep):
                I=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_i1)+tf.matmul(self.state,self.weight_i2)+self.bias_i)
                F=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_f1)+tf.matmul(self.state,self.weight_f2)+self.bias_f)
                O=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_o1)+tf.matmul(self.state,self.weight_o2)+self.bias_o)
                C_=tf.nn.tanh(tf.matmul(data[:][:,j],self.weight_c1)+tf.matmul(self.state,self.weight_c2)+self.bias_c)
                C=I*C_+self.C*F
                output=O*tf.nn.tanh(C)
                self.output_list.append(output)
                self.C=C
                self.state=output
            if self.return_sequence==True:
                output=tf.stack(self.output_list,axis=1)
                self.output_list=[]
                return output
            else:
                self.output_list=[]
                return output
        else:
            for j in range(self.timestep):
                I=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_i1)+tf.matmul(self.state,self.weight_i2))
                F=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_f1)+tf.matmul(self.state,self.weight_f2))
                O=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_o1)+tf.matmul(self.state,self.weight_o2))
                C_=tf.nn.tanh(tf.matmul(data[:][:,j],self.weight_c1)+tf.matmul(self.state,self.weight_c2))
                C=I*C_+self.C_*F
                output=O*tf.nn.tanh(C)
                self.output_list.append(output)
                self.C_=C
                self.state=output
            if self.return_sequence==True:
                output=tf.stack(self.output_list,axis=1)
                self.output_list=[]
                return output
            else:
                self.output_list=[]
                return output
