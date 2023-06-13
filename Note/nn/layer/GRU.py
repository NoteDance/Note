import tensorflow as tf
import Note.nn.initializer as i


class GRU:
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',dtype='float32',return_sequence=False,use_bias=True,activation1=tf.nn.sigmoid,activation2=tf.nn.tanh):
        self.weight_r1=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_r2=i.initializer([weight_shape[1],weight_shape[1]],weight_initializer,dtype)
        self.weight_z1=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_z2=i.initializer([weight_shape[1],weight_shape[1]],weight_initializer,dtype)
        self.weight_h1=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_h2=i.initializer([weight_shape[1],weight_shape[1]],weight_initializer,dtype)
        if use_bias==True:
            self.bias_r=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_z=i.initializer([weight_shape[1]],bias_initializer,dtype)
            self.bias_h=i.initializer([weight_shape[1]],bias_initializer,dtype)
        self.output_list=[]
        self.state=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.H=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.return_sequence=return_sequence
        self.use_bias=use_bias
        self.activation1=activation1
        self.activation2=activation2
        if use_bias==True:
            self.param=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2,self.bias_r,self.bias_z,self.bias_h]
        else:
            self.param=[self.weight_r1,self.weight_z1,self.weight_h1,self.weight_r2,self.weight_z2,self.weight_h2]
    
    
    def output(self,data):
        timestep=data.shape[1]
        if self.use_bias==True:
            for j in range(timestep):
                R=self.activation1(tf.matmul(data[:][:,j],self.weight_r1)+tf.matmul(self.state,self.weight_r2)+self.bias_r)
                Z=self.activation1(tf.matmul(data[:][:,j],self.weight_z1)+tf.matmul(self.state,self.weight_z2)+self.bias_z)
                H_=self.activation2(tf.matmul(data[:][:,j],self.weight_h1)+tf.matmul(R*self.H,self.weight_h2)+self.bias_h)
                output=Z*self.H+(1-Z)*H_
                self.output_list.append(output)
                self.H=output
                self.state=output
            if self.return_sequence==True:
                output=tf.stack(self.output_list,axis=1)
                self.output_list=[]
                return output
            else:
                self.output_list=[]
                return output
        else:
            for j in range(timestep):
                R=self.activation1(tf.matmul(data[:][:,j],self.weight_r1)+tf.matmul(self.state,self.weight_r2))
                Z=self.activation1(tf.matmul(data[:][:,j],self.weight_z1)+tf.matmul(self.state,self.weight_z2))
                H_=self.activation2(tf.matmul(data[:][:,j],self.weight_h1)+tf.matmul(R*self.H,self.weight_h2))
                output=Z*self.H+(1-Z)*H_
                self.output_list.append(output)
                self.H=output
                self.state=output
            if self.return_sequence==True:
                output=tf.stack(self.output_list,axis=1)
                self.output_list=[]
                return output
            else:
                self.output_list=[]
                return output