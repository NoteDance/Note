import tensorflow as tf
import numpy as np
import Note.nn.initializer as i


class self_attention:
    def __init__(self,weight_shape,weight_initializer='uniform',dtype='float64'):
        self.qw=i.initializer(weight_shape,weight_initializer,dtype)
        self.kw=i.initializer(weight_shape,weight_initializer,dtype)
        self.vw=i.initializer(weight_shape,weight_initializer,dtype)
        self.weight_list=[self.qw,self.kw,self.vw]
    
    
    def output(self,data,a,mask=None):
        query=tf.einsum('ijk,kl->ijl',data,self.qw)
        key=tf.einsum('ijk,kl->ijl',data,self.kw)
        value=tf.einsum('ijk,kl->ijl',data,self.vw)
        query=tf.reshape(query,shape=[query.shape[0],query.shape[1],a,int(data.shape[2])/a])
        key=tf.reshape(key,shape=[key.shape[0], key.shape[1],a,int(data.shape[2])/a])
        value=tf.reshape(value,shape=[value.shape[0],value.shape[1],a,int(data.shape[2])/a])
        key=tf.transpose(key,perm=[0,2,3,1])
        value=tf.transpose(value,perm=[0,2,1,3])
        scores=tf.matmul(query,key)/np.sqrt(int(data.shape[2])/a)
        if mask is not None:
            scores+=mask*-1e9
        attention_weights=tf.nn.softmax(scores,axis=-1)
        output=tf.matmul(attention_weights,value)
        output=tf.transpose(output,perm=[0,2,1,3])
        output=tf.reshape(output,shape=[output.shape[0],output.shape[1],-1])
        return output,attention_weights