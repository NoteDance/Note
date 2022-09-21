import tensorflow as tf
import numpy as np


def attention(en_h,de_h,attention_w1,attention_w2,attention_w3,score_en_h=None):
    if type(en_h)==list:
        stack_en_h=tf.stack(en_h,axis=1)
    else:
        stack_en_h=en_h
    if score_en_h==None:
        score=tf.einsum('ijk,kl->ijl',tf.nn.tanh(tf.einsum('ijk,kl->ijl',stack_en_h,attention_w1)+tf.expand_dims(tf.matmul(de_h,attention_w2),axis=1)),attention_w3)
    else:
        score=tf.einsum('ijk,kl->ijl',tf.nn.tanh(score_en_h+tf.expand_dims(tf.matmul(de_h,attention_w2),axis=1)),attention_w3)
    attention_weights=tf.nn.softmax(score,axis=1)
    context_vector=tf.reduce_sum(attention_weights*stack_en_h,axis=1)
    if score_en_h==None:
        return tf.einsum('ijk,kl->ijl',stack_en_h,attention_w1),context_vector
    else:
        return context_vector
    

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
            qk=qk/np.sqrt(int(data.shape[2])/a)
            if mask!=False:
                qk=qk*mask
            softmax=tf.nn.softmax(qk,axis=1)
            softmax=tf.reshape(softmax,shape=[-1,1])
            for k in range(query.shape[1]):
                _value[j][k]=tf.reduce_sum(softmax[k]*value[j,:,i,:],aixs=0)
        output.append(_value)
    output=tf.concat(output,axis=2)
    return output
