import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class RNN:
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zeros',activation=None,dtype='float32',return_sequence=False,use_bias=True):
        self.weight_i=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_s=i.initializer([weight_shape[1],weight_shape[1]],weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([weight_shape[1]],bias_initializer,dtype)
        self.output_list=[]
        self.state=tf.zeros(shape=[1,weight_shape[1]],dtype=dtype)
        self.activation=activation_dict[activation]
        self.return_sequence=return_sequence
        self.use_bias=use_bias
        if use_bias==True:
            self.param=[self.weight_i,self.weight_s,self.bias]
        else:
            self.param=[self.weight_i,self.weight_s]
    
    
    def output(self,data):
        timestep=data.shape[1]
        if self.use_bias==True:
            for j in range(timestep):
                output=self.activation(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
                self.output_list.append(output)
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
                output=self.activation(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s))
                self.output_list.append(output)
                self.state=output
            if self.return_sequence==True:
                output=tf.stack(self.output_list,axis=1)
                self.output_list=[]
                return output
            else:
                self.output_list=[]
                return output