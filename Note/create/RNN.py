import tensorflow as tf


def rnn(data,weight,bias,stack=True):
    h=[]
    if len(data.shape)==3:
        X=tf.einsum('ijk,kl->ijl',data,weight)+bias
    else:
        X=tf.matmul(data,weight)+bias
    for i in range(int(data.shape[1])):
        if i==0:
            h.append(tf.nn.tanh(X[:][:,i]+bias))
        else:
            h.append(tf.nn.tanh(tf.matmul(h[i-1],weight)+X[:][:,i]+bias))
    if stack==True:
        return tf.stack(h,axis=1)
    else:
        return h
    
    
def lstm(data,weight,bias,stack=True):
    C=[]
    h=[]
    if len(data.shape)==3:
        fx=tf.einsum('ijk,kl->ijl',data,weight[0])
        ix=tf.einsum('ijk,kl->ijl',data,weight[1])
        ox=tf.einsum('ijk,kl->ijl',data,weight[2])
        cx=tf.einsum('ijk,kl->ijl',data,weight[3])
    else:
        fx=tf.matmul(data,weight[0])
        ix=tf.matmul(data,weight[1])
        ox=tf.matmul(data,weight[2])
        cx=tf.matmul(data,weight[3])
    for i in range(int(data.shape[1])):
        if i==0:
            f=tf.nn.sigmoid(fx[:][:,i]+bias[0])
            i=tf.nn.sigmoid(ix[:][:,i]+bias[1])
            o=tf.nn.sigmoid(ox[:][:,i]+bias[2])
            c=tf.nn.tanh(cx[:][:,i]+bias[3])
            C.append(i*c)
            h.append(o*tf.nn.tanh(C[i]))
        else:
            f=tf.nn.sigmoid(fx[:][:,i]+tf.matmul(h[i-1],weight[0])+bias[0])
            i=tf.nn.sigmoid(ix[:][:,i]+tf.matmul(h[i-1],weight[1])+bias[1])
            o=tf.nn.sigmoid(ox[:][:,i]+tf.matmul(h[i-1],weight[2])+bias[2])
            c=tf.nn.tanh(cx[:][:,i]+tf.matmul(h[i-1],weight[3])+bias[3])
            C.append(f*C[i-1]+i*c)
            h.append(o*tf.nn.tanh(C[i]))
    if stack==True:
        return tf.stack(h,axis=1)
    else:
        return h


def gru(data,weight,bias,stack=True):
    h=[]
    if len(data.shape)==3:
        ux=tf.einsum('ijk,kl->ijl',data,weight[0])
        rx=tf.einsum('ijk,kl->ijl',data,weight[1])
        cx=tf.einsum('ijk,kl->ijl',data,weight[2])
    else:
        ux=tf.matmul(data,weight[0])
        rx=tf.matmul(data,weight[1])
        cx=tf.matmul(data,weight[2]) 
    for i in range(int(data.shape[1])):
        if i==0:
            u=tf.nn.sigmoid(ux[:][:,i]+bias[0])
            r=tf.nn.sigmoid(rx[:][:,i]+bias[1])
            c=tf.nn.tanh(cx[:][:,i]+bias[2])
            h.append((1-u)*c)
        else:
            u=tf.nn.sigmoid(ux[:][:,i]+tf.matmul(h[i-1],weight[0])+bias[0])
            r=tf.nn.sigmoid(rx[:][:,i]+tf.matmul(h[i-1],weight[1])+bias[1])
            c=tf.nn.tanh(cx[:][:,i]+tf.matmul(r*h[i-1],weight[2])+bias[2])
            h.append(u*h[i-1]+(1-u)*c)
    if stack==True:
        return tf.stack(h,axis=1)
    else:
        return h


def m_relugru(data,weight,bias,stack=True):
    h=[]
    if len(data.shape)==3:
        ux=tf.einsum('ijk,kl->ijl',data,weight[0])
        cx=tf.einsum('ijk,kl->ijl',data,weight[1])
    else:
        ux=tf.matmul(data,weight[0])
        cx=tf.matmul(data,weight[1])
    for i in range(int(data.shape[1])):
        if i==0:
            u=tf.nn.sigmoid(ux[:][:,i]+bias[0])
            c=tf.nn.relu(cx[:][:,i]+bias[1])
            c-=tf.reduce_mean(c,axis=0)
            c/=tf.math.reduce_std(c,axis=0)
            h.append((1-u)*c)
        else:
            u=tf.nn.sigmoid(ux[:][:,i]+tf.matmul(h[i-1],weight[0])+bias[0])
            c=tf.nn.relu(cx[:][:,i]+tf.matmul(h[i-1],weight[1])+bias[1])
            c-=tf.reduce_mean(c,axis=0)
            c/=tf.math.reduce_std(c,axis=0)
            h.append(u*h[i-1]+(1-u)*c)
    if stack==True:
        return tf.stack(h,axis=1)
    else:
        return h


def embedding(data,embedding_w,embedding_b):
    return tf.einsum('ijk,kl->ijl',data,embedding_w)+embedding_b


def attention(en_h,de_h,attention_w1,attention_w2,attention_w3):
    if type(en_h)==list:
        stack_en_h=tf.stack(en_h,axis=1)
    if type(de_h)==tf.Tensor:
        length=int(de_h.shape[1])
    else:
        length=len(de_h)
    attention_vector=[]
    context_vector=[]
    for i in range(length):
        if type(de_h)==tf.Tensor:
            de_h=de_h[:][:,i]
        else:
            de_h=de_h[i]
        score=tf.einsum('ijk,kl->ijl',tf.nn.tanh(tf.expand_dims(tf.matmul(de_h,attention_w1),axis=1)+tf.einsum('ijk,kl->ijl',stack_en_h,attention_w2)),attention_w3)
        score=tf.reduce_mean(score,axis=2)
        score=tf.expand_dims(score,axis=2)
        attention_weights=tf.nn.softmax(score,axis=1)
        context_vector.append(tf.reduce_mean(attention_weights*stack_en_h,axis=1))
        attention_vector.append(tf.concat([de_h,context_vector[i]],axis=1))
    return tf.stack(attention_vector,axis=1)
