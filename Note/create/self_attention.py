import tensorflow as tf
import numpy as np


def self_attention(data,qw,kw,vw,a,mask=False):
    query=tf.einsum('ijk,kl->ijl',data,qw)
    key=tf.einsum('ijk,kl->ijl',data,kw)
    value=tf.einsum('ijk,kl->ijl',data,vw)
    query=tf.reshape(query,shape=[query.shape[0],query.shape[1],a,int(data.shape[2])/a])
    key=tf.reshape(key,shape=[int(data.shape[2])/a,a,key.shape[1],key.shape[0]])
    value=tf.reshape(value,shape=[value.shape[0],value.shape[1],a,int(data.shape[2])/a])
    _value=tf.zeros(shape=[value.shape[0],value.shape[1]])
    output=[]
    if mask!=False:
        mask=tf.constant(np.triu(np.zeros([query.shape[1],query.shape[1]])-1e10))
    for i in range(a):
        for j in range(query.shape[0]):
            qk=tf.matmul(query[j,:,i,:],key[:,i,:,j])
            qk=qk/a
            if mask!=False:
                qk=qk*mask
            softmax=tf.nn.softmax(qk,axis=1)
            softmax=tf.reshape(softmax,shape=[-1,1])
            for k in range(query.shape[1]):
                _value[j][k]=tf.reduce_sum(softmax[k]*value[j,:,i,:],aixs=0)
        output.append(_value)
    output=tf.concat(output,axis=2)
    return output