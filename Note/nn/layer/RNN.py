import tensorflow as tf
import Note.nn.initializer as i


class RNN:
    def __init__(self,weight_shape,timestep,weight_initializer='uniform',bias_initializer='zero',activation=None,dtype='float64',return_sequence=False,use_bias=True):
        self.weight_i=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_s=i.initializer(weight_shape,weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([weight_shape[1]],bias_initializer,dtype)
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
            if self.activation=='sigmoid':
                for j in range(self.timestep):
                    output=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    output=tf.stack(self.output_list,axis=1)
                    self.output_list=[]
                    return output
                else:
                    self.output_list=[]
                    return output
            elif self.activation=='tanh':
                for j in range(self.timestep):
                    output=tf.nn.tanh(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    output=tf.stack(self.output_list,axis=1)
                    self.output_list=[]
                    return output
                else:
                    self.output_list=[]
                    return output
            elif self.activation=='relu':
                for j in range(self.timestep):
                    output=tf.nn.relu(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    output=tf.stack(self.output_list,axis=1)
                    self.output_list=[]
                    return output
                else:
                    self.output_list=[]
                    return output
            elif self.activation=='elu':
                for j in range(self.timestep):
                    output=tf.nn.elu(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s)+self.bias)
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
            if self.activation=='sigmoid':
                for j in range(self.timestep):
                    output=tf.nn.sigmoid(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s))
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    output=tf.stack(self.output_list,axis=1)
                    self.output_list=[]
                    return output
                else:
                    self.output_list=[]
                    return output
            elif self.activation=='tanh':
                for j in range(self.timestep):
                    output=tf.nn.tanh(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s))
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    output=tf.stack(self.output_list,axis=1)
                    self.output_list=[]
                    return output
                else:
                    self.output_list=[]
                    return output
            elif self.activation=='relu':
                for j in range(self.timestep):
                    output=tf.nn.relu(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s))
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    output=tf.stack(self.output_list,axis=1)
                    self.output_list=[]
                    return output
                else:
                    self.output_list=[]
                    return output
            elif self.activation=='elu':
                for j in range(self.timestep):
                    output=tf.nn.elu(tf.matmul(data[:][:,j],self.weight_i)+tf.matmul(self.state,self.weight_s))
                    self.output_list.append(output)
                    self.state=output
                if self.return_sequence==True:
                    output=tf.stack(self.output_list,axis=1)
                    self.output_list=[]
                    return output
                else:
                    self.output_list=[]
                    return output
