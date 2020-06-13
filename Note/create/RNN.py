import tensorflow as tf


def rnn(data,weight,bias):
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
    return tf.stack(h,axis=1)

    
    
def lstm(data,weight_x,weight_h,bias):
    C=[]
    h=[]
    if len(data.shape)==3:
        fx=tf.einsum('ijk,kl->ijl',data,weight_x[0])
        ix=tf.einsum('ijk,kl->ijl',data,weight_x[1])
        ox=tf.einsum('ijk,kl->ijl',data,weight_x[2])
        cx=tf.einsum('ijk,kl->ijl',data,weight_x[3])
    else:
        fx=tf.matmul(data,weight_x[0])
        ix=tf.matmul(data,weight_x[1])
        ox=tf.matmul(data,weight_x[2])
        cx=tf.matmul(data,weight_x[3])
    for i in range(int(data.shape[1])):
        if i==0:
            f=tf.nn.sigmoid(fx[:][:,i]+bias[0])
            i=tf.nn.sigmoid(ix[:][:,i]+bias[1])
            o=tf.nn.sigmoid(ox[:][:,i]+bias[2])
            c=tf.nn.tanh(cx[:][:,i]+bias[3])
            C.append(i*c)
            h.append(o*tf.nn.tanh(C[i]))
        else:
            f=tf.nn.sigmoid(fx[:][:,i]+tf.matmul(h[i-1],weight_h[0])+bias[0])
            i=tf.nn.sigmoid(ix[:][:,i]+tf.matmul(h[i-1],weight_h[1])+bias[1])
            o=tf.nn.sigmoid(ox[:][:,i]+tf.matmul(h[i-1],weight_h[2])+bias[2])
            c=tf.nn.tanh(cx[:][:,i]+tf.matmul(h[i-1],weight_h[3])+bias[3])
            C.append(f*C[i-1]+i*c)
            h.append(o*tf.nn.tanh(C[i]))
    return tf.stack(h,axis=1)


def gru(data,weight_x,weight_h,bias):
    h=[]
    if len(data.shape)==3:
        ux=tf.einsum('ijk,kl->ijl',data,weight_x[0])
        rx=tf.einsum('ijk,kl->ijl',data,weight_x[1])
        cx=tf.einsum('ijk,kl->ijl',data,weight_x[2])
    else:
        ux=tf.matmul(data,weight_x[0])
        rx=tf.matmul(data,weight_x[1])
        cx=tf.matmul(data,weight_x[2]) 
    for i in range(int(data.shape[1])):
        if i==0:
            u=tf.nn.sigmoid(ux[:][:,i]+bias[0])
            r=tf.nn.sigmoid(rx[:][:,i]+bias[1])
            c=tf.nn.tanh(cx[:][:,i]+bias[2])
            h.append((1-u)*c)
        else:
            u=tf.nn.sigmoid(ux[:][:,i]+tf.matmul(h[i-1],weight_h[0])+bias[0])
            r=tf.nn.sigmoid(rx[:][:,i]+tf.matmul(h[i-1],weight_h[1])+bias[1])
            c=tf.nn.tanh(cx[:][:,i]+tf.matmul(r*h[i-1],weight_h[2])+bias[2])
            h.append(u*h[i-1]+(1-u)*c)
    return tf.stack(h,axis=1)


def m_relugru(data,weight_x,weight_h,bias):
    h=[]
    if len(data.shape)==3:
        ux=tf.einsum('ijk,kl->ijl',data,weight_x[0])
        cx=tf.einsum('ijk,kl->ijl',data,weight_x[1])
    else:
        ux=tf.matmul(data,weight_x[0])
        cx=tf.matmul(data,weight_x[1])
    for i in range(int(data.shape[1])):
        if i==0:
            u=tf.nn.sigmoid(ux[:][:,i]+bias[0])
            c=tf.nn.relu(cx[:][:,i]+bias[1])
            c-=tf.reduce_mean(c,axis=0)
            c/=tf.math.reduce_std(c,axis=0)
            h.append((1-u)*c)
        else:
            u=tf.nn.sigmoid(ux[:][:,i]+tf.matmul(h[i-1],weight_h[0])+bias[0])
            c=tf.nn.relu(cx[:][:,i]+tf.matmul(h[i-1],weight_h[1])+bias[1])
            c-=tf.reduce_mean(c,axis=0)
            c/=tf.math.reduce_std(c,axis=0)
            h.append(u*h[i-1]+(1-u)*c)
    return tf.stack(h,axis=1)


def rnn_weight(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)


def rnn_bias(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)


def lstm_weight_x(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        fg_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        ig_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        og_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        fg_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        ig_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        og_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[2])
        cltm_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return fg_weight_x,ig_weight_x,og_weight_x,cltm_weight_x


def lstm_weight_h(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        fg_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        ig_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        og_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        fg_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        ig_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        og_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[2])
        cltm_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return fg_weight_h,ig_weight_h,og_weight_h,cltm_weight_h


def lstm_bias(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        fg_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        ig_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        og_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        fg_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        ig_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        og_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[2])
        cltm_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return fg_bias,ig_bias,og_bias,cltm_bias


def gru_weight_x(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        ug_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        rg_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        ug_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        rg_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        cltm_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return ug_weight_x,rg_weight_x,cltm_weight_x


def gru_weight_h(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        ug_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        rg_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        ug_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        rg_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        cltm_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return ug_weight_h,rg_weight_h,cltm_weight_h


def gru_bias(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        ug_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        rg_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        ug_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        rg_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        cltm_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return ug_bias,rg_bias,cltm_bias


def m_relugru_weight_x(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        ug_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        rg_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        ug_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        rg_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        cltm_weight_x=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return ug_weight_x,rg_weight_x,cltm_weight_x


def m_relugru_weight_h(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        ug_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        rg_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        ug_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        rg_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        cltm_weight_h=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return ug_weight_h,rg_weight_h,cltm_weight_h


def m_relugru_bias(shape,mean=0,stddev=0.07,dtype=tf.float32,name=None):
    if name==None:
        ug_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        rg_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
        cltm_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name)
    else:
        ug_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[0])
        rg_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[1])
        cltm_bias=tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=dtype),name=name[3])
    return ug_bias,rg_bias,cltm_bias


def embedding(data,embedding_w,embedding_b):
    return tf.einsum('ijk,kl->ijl',data,embedding_w)+embedding_b


def attention(en_h,de_h,attention_w1,attention_w2,attention_w3):
    if type(en_h)==list:
        stack_en_h=tf.stack(en_h,axis=1)
    length=int(de_h.shape[1])
    attention_vector=[]
    context_vector=[]
    for i in range(length):
        if type(de_h)==tf.Tensor:
            de_h=de_h[:][:,i]
        else:
            de_h=de_h[i]
        score=tf.einsum('ijk,kl->ijl',tf.nn.tanh(tf.expand_dims(tf.matmul(de_h,attention_w1),axis=1)+tf.einsum('ijk,kl->ijl',stack_en_h,attention_w2)),attention_w3)
        attention_weights=tf.nn.softmax(score,axis=1)
        context_vector.append(tf.reduce_sum(attention_weights*stack_en_h,axis=1))
    return tf.stack(context_vector,axis=1)
