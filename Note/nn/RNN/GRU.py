import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class GRU:
    def __init__(self,train_data=None,train_labels=None,test_data=None,test_labels=None):
        self.graph=tf.Graph()
        self.train_data=train_data
        self.train_labels=train_labels
        self.test_data=train_data
        self.test_labels=train_labels
        with self.graph.as_default():
            if type(train_data)==np.ndarray:
                self.shape0=train_data.shape[0]
                self.data_shape=train_data.shape
                self.labels_shape=train_labels.shape
                self.data=tf.placeholder(dtype=train_data.dtype,shape=[None,None,None],name='data')
                if len(self.labels_shape)==3:
                    self.labels=tf.placeholder(dtype=train_labels.dtype,shape=[None,None,None],name='labels')
                elif len(self.labels_shape)==2:
                    self.labels=tf.placeholder(dtype=train_labels.dtype,shape=[None,None],name='labels')
                self.train_data_dtype=train_data.dtype
                self.train_labels_dtype=np.int32
        self.hidden=None
        self.pattern=None
        self.layers=None
        self.embedding_w=None
        self.embedding_b=None
        self.ug_weight_x=None
        self.ug_weight_h=None
        self.rg_weight_x=None
        self.rg_weight_h=None
        self.cltm_weight_x=None
        self.cltm_weight_h=None
        self.weight_o=None
        self.ug_bias=None        
        self.rg_bias=None
        self.cltm_bias=None
        self.bias_o=None
        self.h=[]
        self.output=None
        self.last_embedding_w=None
        self.last_embedding_b=None
        self.last_ug_weight_x=None
        self.last_ug_weight_h=None
        self.last_rg_weight_x=None
        self.last_rg_weight_h=None
        self.last_cltm_weight_x=None
        self.last_cltm_weight_h=None
        self.last_weight_o=None
        self.last_ug_bias=None
        self.last_rg_bias=None
        self.last_cltm_bias=None
        self.last_bias_o=None
        self.batch=None
        self.epoch=0
        self.dropout=None
        self.lr=None
        self.optimizer=None
        self.train_loss=None
        self.train_accuracy=None
        self.train_loss_list=[]
        self.train_accuracy_list=[]
        self.test_loss=None
        self.test_accuracy=None
        self.test_loss_list=[]
        self.test_accuracy_list=[]
        self.total_epoch=0
        self.time=0
        self.total_time=0
        self.processor='/gpu:0'
    
    
    def embedding(self,d,mean=0.07,stddev=0.07,dtype=tf.float32):
        self.embedding_w=self.weight_init([self.data_shape[2],d],mean=mean,stddev=stddev,dtype=dtype,name='embedding_w')
        self.embedding_b=self.bias_init([d],mean=mean,stddev=stddev,dtype=dtype,name='embedding_b')
        return
    
    
    def weight_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
            
    
    def bias_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)

    
    def structure(self,hidden,pattern,layers=None,predicate=False,mean=0,stddev=0.07,dtype=tf.float32):
        with self.graph.as_default():
            self.continue_train=False
            self.flag=None
            self.end_flag=False
            self.test_flag=False
            self.train_loss_list.clear()
            self.train_accuracy_list.clear()
            self.hidden=hidden
            self.pattern=pattern
            self.layers=layers
            self.predicate=predicate
            self.epoch=0
            self.dtype=dtype
            self.total_epoch=0
            self.time=0
            self.total_time=0
            with tf.name_scope('parameter_initialization'):
                if self.layers!=None:
                    self.ug_weight_x=[]
                    self.ug_weight_h=[]
                    self.ug_bias=[]
                    self.rg_weight_x=[]
                    self.rg_weight_h=[]
                    self.rg_bias=[]
                    self.cltm_weight_x=[]
                    self.cltm_weight_h=[]
                    self.cltm_bias=[]
                    for i in range(self.layers):
                        if i==0:
                            self.ug_weight_x.append(self.weight_init([self.data_shape[2],self.hidden],mean=mean,stddev=stddev,name='ug_weight_x{}'.format(i+1)))
                            self.ug_weight_h.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='ug_weight_h{}'.format(i+1)))
                            self.ug_bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='ug_bias{}'.format(i+1)))
                            self.rg_weight_x.append(self.weight_init([self.data_shape[2],self.hidden],mean=mean,stddev=stddev,name='rg_weight_x{}'.format(i+1)))
                            self.rg_weight_h.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='rg_weight_h{}'.format(i+1)))
                            self.rg_bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='rg_bias{}'.format(i+1)))
                            self.cltm_weight_x.append(self.weight_init([self.data_shape[2],self.hidden],mean=mean,stddev=stddev,name='cltm_weight_x{}'.format(i+1)))
                            self.cltm_weight_h.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='cltm_weight_h{}'.format(i+1)))
                            self.cltm_bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='cltm_bias{}'.format(i+1)))
                        else:
                            self.ug_weight_x.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='ug_weight_x{}'.format(i+1)))
                            self.ug_weight_h.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='ug_weight_h{}'.format(i+1)))
                            self.ug_bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='ug_bias{}'.format(i+1)))
                            self.rg_weight_x.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='rg_weight_x{}'.format(i+1)))
                            self.rg_weight_h.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='rg_weight_h{}'.format(i+1)))
                            self.rg_bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='rg_bias{}'.format(i+1)))
                            self.cltm_weight_x.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='cltm_weight_x{}'.format(i+1)))
                            self.cltm_weight_h.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='cltm_weight_h{}'.format(i+1)))
                            self.cltm_bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='cltm_bias{}'.format(i+1)))
                    if len(self.labels_shape)==3:
                        self.weight_o=self.weight_init([self.hidden,self.labels_shape[2]],mean=mean,stddev=stddev,name='weight_o')
                        self.bias_o=self.bias_init([self.labels_shape[2]],mean=mean,stddev=stddev,name='bias_o')
                    elif len(self.labels_shape)==2:
                        self.weight_o=self.weight_init([self.hidden,self.labels_shape[1]],mean=mean,stddev=stddev,name='weight_o')
                        self.bias_o=self.bias_init([self.labels_shape[1]],mean=mean,stddev=stddev,name='bias_o')
                    return
                self.ug_weight_x=self.weight_init([self.data_shape[2],self.hidden],mean=mean,stddev=stddev,name='ug_weight_x')
                self.ug_weight_h=self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='ug_weight_h')
                self.ug_bias=self.bias_init([self.hidden],mean=mean,stddev=stddev,name='ug_bias')
                self.rg_weight_x=self.weight_init([self.data_shape[2],self.hidden],mean=mean,stddev=stddev,name='rg_weight_x')
                self.rg_weight_h=self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='rg_weight_h')
                self.rg_bias=self.bias_init([self.hidden],mean=mean,stddev=stddev,name='rg_bias')
                self.cltm_weight_x=self.weight_init([self.data_shape[2],self.hidden],mean=mean,stddev=stddev,name='cltm_weight_x')
                self.cltm_weight_h=self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='cltm_weight_h')
                self.cltm_bias=self.bias_init([self.hidden],mean=mean,stddev=stddev,name='cltm_bias')
                if len(self.labels_shape)==3:
                    self.weight_o=self.weight_init([self.hidden,self.labels_shape[2]],mean=mean,stddev=stddev,name='weight_o')
                    self.bias_o=self.bias_init([self.labels_shape[2]],mean=mean,stddev=stddev,name='bias_o')
                elif len(self.labels_shape)==2:
                    self.weight_o=self.weight_init([self.hidden,self.labels_shape[1]],mean=mean,stddev=stddev,name='weight_o')
                    self.bias_o=self.bias_init([self.labels_shape[1]],mean=mean,stddev=stddev,name='bias_o')
                return
            
                
    def forward_propagation(self,data,labels=None,use_nn=False):
        with self.graph.as_default():
            if self.layers!=None:
                processor=[]
                for i in range(self.layers):
                    if type(self.processor)==str:
                        processor.append(self.processor)
                    else:
                        processor.append(self.processor[i])
            else:
                processor=self.processor
            self.output=None
            if use_nn==False:  
                embedding_w=self.embedding_w
                embedding_b=self.embedding_b
                ug_weight_x=self.fg_weight_x
                ug_weight_h=self.fg_weight_h
                rg_weight_x=self.ig_weight_x
                rg_weight_h=self.ig_weight_h
                cltm_weight_x=self.cltm_weight_x
                cltm_weight_h=self.cltm_weight_h
                ug_bias=self.fg_bias
                rg_bias=self.ig_bias
                cltm_bias=self.cltm_bias
                weight_o=self.weight_o
                bias_o=self.bias_o
            else:     
                embedding_w=tf.constant(self.last_embedding_w)
                embedding_b=tf.constant(self.last_embedding_b)
                if self.layers!=None:
                    ug_weight_x=[]
                    ug_weight_h=[]
                    rg_weight_x=[]
                    rg_weight_h=[]
                    cltm_weight_x=[]
                    cltm_weight_h=[]
                    ug_bias=[]
                    rg_bias=[]
                    cltm_bias=[]
                    for i in range(self.layers):
                        ug_weight_x.append(tf.constant(self.last_fg_weight_x[i]))
                        ug_weight_h.append(tf.constant(self.last_fg_weight_h[i]))
                        rg_weight_x.append(tf.constant(self.last_ig_weight_x[i]))
                        rg_weight_h.append(tf.constant(self.last_ig_weight_h[i]))
                        cltm_weight_x.append(tf.constant(self.last_cltm_weight_x[i]))
                        cltm_weight_h.append(tf.constant(self.last_cltm_weight_h[i]))
                        ug_bias.append(tf.constant(self.last_fg_bias[i]))
                        rg_bias.append(tf.constant(self.last_ig_bias[i]))
                        cltm_bias.append(tf.constant(self.last_cltm_bias[i]))
                    weight_o=tf.constant(self.weight_o)
                    bias_o=tf.constant(self.bias_o)
                else:
                    ug_weight_x=tf.constant(self.last_fg_weight_x)
                    ug_weight_h=tf.constant(self.last_fg_weight_h)
                    rg_weight_x=tf.constant(self.last_ig_weight_x)
                    rg_weight_h=tf.constant(self.last_ig_weight_h)
                    cltm_weight_x=tf.constant(self.last_cltm_weight_x)
                    cltm_weight_h=tf.constant(self.last_cltm_weight_h)
                    ug_bias=tf.constant(self.last_fg_bias)
                    rg_bias=tf.constant(self.last_ig_bias)
                    cltm_bias=tf.constant(self.last_cltm_bias)
                    weight_o=tf.constant(self.last_weight_o)
                    bias_o=tf.constant(self.last_bias_o)
            with tf.name_scope('forward_propagation'):
                if self.layers!=None:
                    ux=[]
                    rx=[]
                    cx=[]
                    u=[x for x in range(self.layers)]
                    r=[x for x in range(self.layers)]
                    c=[x for x in range(self.layers)]
                    self.h=[x for x in range(self.layers)]
                    for j in range(self.layers):
                        with tf.device(processor[j]):
                            self.h[j]=[]
                            if j==0:
                                data=tf.einsum('ijk,kl->ijl',data,embedding_w)+embedding_b
                            if j==0:
                                ux.append(tf.einsum('ijk,kl->ijl',data,ug_weight_x[j]))
                                rx.append(tf.einsum('ijk,kl->ijl',data,rg_weight_x[j]))
                                cx.append(tf.einsum('ijk,kl->ijl',data,cltm_weight_x[j]))
                                if self.pattern=='1n':
                                    for k in range(self.labels_shape[1]):
                                        if k==0:
                                            u[j]=tf.nn.sigmoid(ux[j]+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+cltm_bias[j])
                                            self.h[j].append((1-u[j])*c[j])
                                        else:
                                            u[j]=tf.nn.sigmoid(ux[j]+tf.matmul(self.h[j][k-1],ug_weight_h[j])+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+tf.matmul(self.h[j][k-1],rg_weight_h[j])+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+tf.matmul(r[j]*self.h[j][k-1],cltm_weight_h[j])+cltm_bias[j])
                                            self.h[j].append(u[j]*self.h[j][k-1]+(1-u[j])*c[j])
                                    self.h[j]=tf.stack(self.h[j],axis=1)
                                elif self.pattern=='n1' or self.predicate==True:
                                    for k in range(self.data_shape[1]):
                                        if k==0:
                                            u[j]=tf.nn.sigmoid(ux[j]+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+cltm_bias[j])
                                            self.h[j].append((1-u[j])*c[j])
                                        else:
                                            u[j]=tf.nn.sigmoid(ux[j]+tf.matmul(self.h[j][k-1],ug_weight_h[j])+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+tf.matmul(self.h[j][k-1],rg_weight_h[j])+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+tf.matmul(r[j]*self.h[j][k-1],cltm_weight_h[j])+cltm_bias[j])
                                            self.h[j].append(u[j]*self.h[j][k-1]+(1-u[j])*c[j])
                                    self.h[j]=tf.stack(self.h[j],axis=1)
                                elif self.pattern=='nn':
                                    for k in range(self.data_shape[1]):
                                        if k==0:
                                            u[j]=tf.nn.sigmoid(ux[j]+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+cltm_bias[j])
                                            self.h[j].append((1-u[j])*c[j])
                                        else:
                                            u[j]=tf.nn.sigmoid(ux[j]+tf.matmul(self.h[j][k-1],ug_weight_h[j])+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+tf.matmul(self.h[j][k-1],rg_weight_h[j])+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+tf.matmul(r[j]*self.h[j][k-1],cltm_weight_h[j])+cltm_bias[j])
                                            self.h[j].append(u[j]*self.h[j][k-1]+(1-u[j])*c[j])
                                    self.h[j]=tf.stack(self.h[j],axis=1)
                            else:
                                ux.append(tf.einsum('ijk,kl->ijl',self.h[j-1],ug_weight_x[j]))
                                rx.append(tf.einsum('ijk,kl->ijl',self.h[j-1],rg_weight_x[j]))
                                cx.append(tf.einsum('ijk,kl->ijl',self.h[j-1],cltm_weight_x[j]))
                                if self.pattern=='1n':
                                    for k in range(self.labels_shape[1]):
                                        if k==0:
                                            u[j]=tf.nn.sigmoid(ux[j]+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+cltm_bias[j])
                                            self.h[j].append((1-u[j])*c[j])
                                        else:
                                            u[j]=tf.nn.sigmoid(ux[j]+tf.matmul(self.h[j][k-1],ug_weight_h[j])+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+tf.matmul(self.h[j][k-1],rg_weight_h[j])+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+tf.matmul(r[j]*self.h[j][k-1],cltm_weight_h[j])+cltm_bias[j])
                                            self.h[j].append(u[j]*self.h[j][k-1]+(1-u[j])*c[j])
                                    if j==self.layers-1:
                                        self.output=tf.einsum('ijk,kl->ijl',tf.stack(self.h[-1],axis=1),weight_o)+bias_o
                                    else:
                                        self.h[j]=tf.stack(self.h[j],axis=1)
                                elif self.pattern=='n1' or self.predicate==True:
                                    for k in range(self.data_shape[1]):
                                        if k==0:
                                            u[j]=tf.nn.sigmoid(ux[j]+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+cltm_bias[j])
                                            self.h[j].append((1-u[j])*c[j])
                                        else:
                                            u[j]=tf.nn.sigmoid(ux[j]+tf.matmul(self.h[j][k-1],ug_weight_h[j])+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+tf.matmul(self.h[j][k-1],rg_weight_h[j])+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+tf.matmul(r[j]*self.h[j][k-1],cltm_weight_h[j])+cltm_bias[j])
                                            self.h[j].append(u[j]*self.h[j][k-1]+(1-u[j])*c[j])
                                    if j==self.layers-1:
                                        for k in range(self.data_shape[1]):
                                            self.output.append(tf.add(tf.matmul(self.h[-1][k],weight_o),bias_o))
                                    else:
                                        self.h[j]=tf.stack(self.h[j],axis=1)
                                elif self.pattern=='nn':
                                    for k in range(self.data_shape[1]):
                                        if k==0:
                                            u[j]=tf.nn.sigmoid(ux[j]+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+cltm_bias[j])
                                            self.h[j].append((1-u[j])*c[j])
                                        else:
                                            u[j]=tf.nn.sigmoid(ux[j]+tf.matmul(self.h[j][k-1],ug_weight_h[j])+ug_bias[j])
                                            r[j]=tf.nn.sigmoid(rx[j]+tf.matmul(self.h[j][k-1],rg_weight_h[j])+rg_bias[j])
                                            c[j]=tf.nn.tanh(cx[j]+tf.matmul(r[j]*self.h[j][k-1],cltm_weight_h[j])+cltm_bias[j])
                                            self.h[j].append(u[j]*self.h[j][k-1]+(1-u[j])*c[j])
                                    if j==self.layers-1:
                                        self.output=tf.einsum('ijk,kl->ijl',tf.stack(self.h[-1],axis=1),weight_o)+bias_o
                                    else:
                                        self.h[j]=tf.stack(self.h[j],axis=1)
                        return
                else:
                    with tf.device(processor):
                        data=tf.einsum('ijk,kl->ijl',data,embedding_w)+embedding_b
                        ux=tf.einsum('ijk,kl->ijl',data,ug_weight_x)
                        rx=tf.einsum('ijk,kl->ijl',data,rg_weight_x)
                        cx=tf.einsum('ijk,kl->ijl',data,cltm_weight_x)
                        if self.pattern=='1n':
                            for j in range(self.labels_shape[1]):
                                if j==0:
                                    u=tf.nn.sigmoid(ux+ug_bias)
                                    r=tf.nn.sigmoid(rx+rg_bias)
                                    c=tf.nn.tanh(cx+cltm_bias)
                                    self.h.append((1-u)*c[j])
                                else:
                                    u=tf.nn.sigmoid(ux+tf.matmul(self.h[j],ug_weight_h)+ug_bias)
                                    r=tf.nn.sigmoid(rx+tf.matmul(self.h[j],rg_weight_h)+rg_bias)
                                    c=tf.nn.tanh(cx+tf.matmul(r*self.h[j],cltm_weight_h)+cltm_bias)
                                    self.h.append(u*self.h[j-1]+(1-u)*c)
                            self.output=tf.einsum('ijk,kl->ijl',tf.stack(self.h,axis=1),weight_o)+bias_o
                        elif self.pattern=='n1' or self.predicate==True:
                            for j in range(self.data_shape[1]):
                                if j==0:
                                    u=tf.nn.sigmoid(ux+ug_bias)
                                    r=tf.nn.sigmoid(rx+rg_bias)
                                    c=tf.nn.tanh(cx+cltm_bias)
                                    self.h.append((1-u)*c[j])
                                else:
                                    u=tf.nn.sigmoid(ux+tf.matmul(self.h[j],ug_weight_h)+ug_bias)
                                    r=tf.nn.sigmoid(rx+tf.matmul(self.h[j],rg_weight_h)+rg_bias)
                                    c=tf.nn.tanh(cx+tf.matmul(r*self.h[j],cltm_weight_h)+cltm_bias)
                                    self.h.append(u*self.h[j-1]+(1-u)*c)
                                self.output.append(tf.add(tf.matmul(self.h[j+1],weight_o),bias_o))
                        elif self.pattern=='nn':
                            for j in range(self.data_shape[1]):
                                if j==0:
                                    u=tf.nn.sigmoid(ux+ug_bias)
                                    r=tf.nn.sigmoid(rx+rg_bias)
                                    c=tf.nn.tanh(cx+cltm_bias)
                                    self.h.append((1-u)*c[j])
                                else:
                                    u=tf.nn.sigmoid(ux+tf.matmul(self.h[j],ug_weight_h)+ug_bias)
                                    r=tf.nn.sigmoid(rx+tf.matmul(self.h[j],rg_weight_h)+rg_bias)
                                    c=tf.nn.tanh(cx+tf.matmul(r*self.h[j],cltm_weight_h)+cltm_bias)
                                    self.h.append(u*self.h[j-1]+(1-u)*c)
                            self.output=tf.einsum('ijk,kl->ijl',tf.stack(self.h,axis=1),weight_o)+bias_o
                        return
            
        
    def train(self,batch=None,epoch=None,optimizer='Adam',lr=0.001,l2=None,test=False,test_batch=None,train_summary_path=None,model_path=None,one=True,continue_train=False,processor=None):
        with self.graph.as_default():
            self.C.clear()
            self.h.clear()
            self.batch=batch
            if batch!=None:
                if batch!=1:
                    random=np.arange(batch)
                else:
                    random=np.arange(self.shape0)
            self.epoch=0
            self.l2=l2
            self.optimizer=optimizer
            self.lr=lr
            self.test_flag=test
            if continue_train!=True:
                if self.continue_train==True:
                    continue_train=True
                else:
                    self.train_loss_list.clear()
                    self.train_accuracy_list.clear()
                    self.test_loss_list.clear()
                    self.test_accuracy_list.clear()
            if self.continue_train==False and continue_train==True:
                self.continue_train=True
            if processor!=None:
                self.processor=processor
            if continue_train==True and self.end_flag==True:
                self.end_flag=False
                self.embedding_w=tf.Variable(self.last_embedding_w,name='embedding_w')
                self.embedding_b=tf.Variable(self.last_embedding_b,name='embedding_b')
                if self.layers!=None:
                    self.ug_weight_x=[]
                    self.ug_weight_h=[]
                    self.rg_weight_x=[]
                    self.rg_weight_h=[]
                    self.cltm_weight_x=[]
                    self.cltm_weight_h=[]
                    self.ug_bias=[]
                    self.rg_bias=[]
                    self.cltm_bias=[]
                    for i in range(self.layers):
                        self.ug_weight_x.append(tf.Variable(self.last_fg_weight_x[i],name='ug_weight_x{}'.format(i+1)))
                        self.ug_weight_h.append(tf.Variable(self.last_fg_weight_h[i],name='ug_weight_h{}'.format(i+1)))
                        self.rg_weight_x.append(tf.Variable(self.last_ig_weight_x[i],name='rg_weight_x{}'.format(i+1)))
                        self.rg_weight_h.append(tf.Variable(self.last_ig_weight_h[i],name='rg_weight_h{}'.format(i+1)))
                        self.cltm_weight_x.append(tf.Variable(self.last_cltm_weight_x[i],name='cltm_weight_x{}'.format(i+1)))
                        self.cltm_weight_h.append(tf.Variable(self.last_cltm_weight_h[i],name='cltm_weight_h{}'.format(i+1)))
                        self.ug_bias.append(tf.Variable(self.last_fg_bias[i],name='ug_bias{}'.format(i+1)))
                        self.rg_bias.append(tf.Variable(self.last_ig_bias[i],name='rg_bias{}'.format(i+1)))
                        self.cltm_bias.append(tf.Variable(self.last_cltm_bias[i],name='cltm_bias{}'.format(i+1)))
                    self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                    self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                else:
                    self.ug_weight_x=tf.Variable(self.last_fg_weight_x,name='ug_weight_x')
                    self.ug_weight_h=tf.Variable(self.last_fg_weight_h,name='ug_weight_h')
                    self.rg_weight_x=tf.Variable(self.last_ig_weight_x,name='rg_weight_x')
                    self.rg_weight_h=tf.Variable(self.last_ig_weight_h,name='rg_weight_h')
                    self.cltm_weight_x=tf.Variable(self.last_cltm_weight_x,name='cltm_weight_x')
                    self.cltm_weight_h=tf.Variable(self.last_cltm_weight_h,name='cltm_weight_h')
                    self.ug_bias=tf.Variable(self.last_fg_bias,name='ug_bias')
                    self.rg_bias=tf.Variable(self.last_ig_bias,name='rg_bias')
                    self.cltm_bias=tf.Variable(self.last_cltm_bias,name='cltm_bias')
                    self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                    self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                self.last_embedding_w=None
                self.last_embedding_b=None
                self.last_ug_weight_x=None
                self.last_ug_weight_h=None
                self.last_rg_weight_x=None
                self.last_rg_weight_h=None
                self.last_cltm_weight_x=None
                self.last_cltm_weight_h=None
                self.last_ug_bias=None
                self.last_rg_bias=None
                self.last_cltm_bias=None
                self.last_weight_o=None
                self.last_bias_o=None
            if continue_train==True and self.flag==1:
                self.flag=0
                self.embedding_w=tf.Variable(self.last_embedding_w,name='embedding_w')
                self.embedding_b=tf.Variable(self.last_embedding_b,name='embedding_b')
                if self.layers!=None:
                    self.ug_weight_x=[]
                    self.ug_weight_h=[]
                    self.rg_weight_x=[]
                    self.rg_weight_h=[]
                    self.cltm_weight_x=[]
                    self.cltm_weight_h=[]
                    self.ug_bias=[]
                    self.rg_bias=[]
                    self.cltm_bias=[]
                    for i in range(self.layers):
                        self.ug_weight_x.append(tf.Variable(self.last_ug_weight_x[i],name='ug_weight_x{}'.format(i+1)))
                        self.ug_weight_h.append(tf.Variable(self.last_ug_weight_h[i],name='ug_weight_h{}'.format(i+1)))
                        self.rg_weight_x.append(tf.Variable(self.last_rg_weight_x[i],name='rg_weight_x{}'.format(i+1)))
                        self.rg_weight_h.append(tf.Variable(self.last_rg_weight_h[i],name='rg_weight_h{}'.format(i+1)))
                        self.cltm_weight_x.append(tf.Variable(self.last_cltm_weight_x[i],name='cltm_weight_x{}'.format(i+1)))
                        self.cltm_weight_h.append(tf.Variable(self.last_cltm_weight_h[i],name='cltm_weight_h{}'.format(i+1)))
                        self.ug_bias.append(tf.Variable(self.last_ug_bias[i],name='ug_bias{}'.format(i+1)))
                        self.rg_bias.append(tf.Variable(self.last_rg_bias[i],name='rg_bias{}'.format(i+1)))
                        self.cltm_bias.append(tf.Variable(self.last_cltm_bias[i],name='cltm_bias{}'.format(i+1)))
                    self.weight_o=tf.Variable(self.weight_o,name='weight_o')
                    self.bias_o=tf.Variable(self.bias_o,name='bias_o')
                else:
                    self.ug_weight_x=tf.Variable(self.last_ug_weight_x,name='ug_weight_x')
                    self.ug_weight_h=tf.Variable(self.last_ug_weight_h,name='ug_weight_h')
                    self.rg_weight_x=tf.Variable(self.last_rg_weight_x,name='rg_weight_x')
                    self.rg_weight_h=tf.Variable(self.last_rg_weight_h,name='rg_weight_h')
                    self.cltm_weight_x=tf.Variable(self.last_cltm_weight_x,name='cltm_weight_x')
                    self.cltm_weight_h=tf.Variable(self.last_cltm_weight_h,name='cltm_weight_h')
                    self.ug_bias=tf.Variable(self.last_ug_bias,name='ug_bias')
                    self.rg_bias=tf.Variable(self.last_rg_bias,name='rg_bias')
                    self.cltm_bias=tf.Variable(self.last_cltm_bias,name='cltm_bias')
                    self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                    self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                self.last_embedding_w=None
                self.last_embedding_b=None
                self.last_ug_weight_x=None
                self.last_ug_weight_h=None
                self.last_rg_weight_x=None
                self.last_rg_weight_h=None
                self.last_cltm_weight_x=None
                self.last_cltm_weight_h=None
                self.last_ug_bias=None
                self.last_rg_bias=None
                self.last_cltm_bias=None
                self.last_weight_o=None
                self.last_bias_o=None
#     －－－－－－－－－－－－－－－forward propagation－－－－－－－－－－－－－－－
            self.forward_propagation(self.data)
#     －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
            with tf.name_scope('train_loss'):
                if self.pattern=='1n':
                    if l2==None:
                        train_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.labels,axis=2),axis=1))
                    else:
                        train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.labels,axis=2),axis=1)
                        if self.layers==None:
                            train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.ug_weight_x**2)+tf.reduce_sum(self.ug_weight_h**2)+tf.reduce_sum(self.rg_weight_x**2)+tf.reduce_sum(self.rg_weight_h**2)+tf.reduce_sum(self.cltm_weight_x**2)+tf.reduce_sum(self.cltm_weight_h**2)+tf.reduce_sum(self.weight_o**2)))
                        else:
                            train_loss=tf.reduce_mean(train_loss+l2/2*(sum([tf.reduce_sum(x**2) for x in self.ug_weight_x])+sum([tf.reduce_sum(x**2) for x in self.ug_weight_h])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_x])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_h])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_x])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_h])+sum([tf.reduce_sum(x**2) for x in self.weight_o])))
                elif self.pattern=='n1' or self.predicate==True:
                    if self.pattern=='n1':
                        if l2==None:
                            train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output[-1],labels=self.labels))
                        else:
                            train_loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output[-1],labels=self.labels)
                            if self.layers==None:
                                train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.ug_weight_x**2)+tf.reduce_sum(self.ug_weight_h**2)+tf.reduce_sum(self.rg_weight_x**2)+tf.reduce_sum(self.rg_weight_h**2)+tf.reduce_sum(self.cltm_weight_x**2)+tf.reduce_sum(self.cltm_weight_h**2)+tf.reduce_sum(self.weight_o**2)))
                            else:
                                train_loss=tf.reduce_mean(train_loss+l2/2*(sum([tf.reduce_sum(x**2) for x in self.ug_weight_x])+sum([tf.reduce_sum(x**2) for x in self.ug_weight_h])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_x])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_h])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_x])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_h])+sum([tf.reduce_sum(x**2) for x in self.weight_o])))
                    else:
                        if l2==None:
                            train_loss=tf.reduce_mean(tf.square(self.output[-1]-tf.expand_dims(self.labels,axis=1)))
                        else:
                            train_loss=tf.square(self.output[-1]-tf.expand_dims(self.labels,axis=1))
                            if self.layers==None:
                                train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.ug_weight_x**2)+tf.reduce_sum(self.ug_weight_h**2)+tf.reduce_sum(self.rg_weight_x**2)+tf.reduce_sum(self.rg_weight_h**2)+tf.reduce_sum(self.cltm_weight_x**2)+tf.reduce_sum(self.cltm_weight_h**2)+tf.reduce_sum(self.weight_o**2)))
                            else:
                                train_loss=tf.reduce_mean(train_loss+l2/2*(sum([tf.reduce_sum(x**2) for x in self.ug_weight_x])+sum([tf.reduce_sum(x**2) for x in self.ug_weight_h])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_x])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_h])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_x])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_h])+sum([tf.reduce_sum(x**2) for x in self.weight_o])))
                elif self.pattern=='nn':
                    if l2==None:
                        train_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.labels,axis=2),axis=1))
                    else:
                        train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.labels,axis=2),axis=1)
                        if self.layers==None:
                            train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.ug_weight_x**2)+tf.reduce_sum(self.ug_weight_h**2)+tf.reduce_sum(self.rg_weight_x**2)+tf.reduce_sum(self.rg_weight_h**2)+tf.reduce_sum(self.cltm_weight_x**2)+tf.reduce_sum(self.cltm_weight_h**2)+tf.reduce_sum(self.weight_o**2)))
                        else:
                            train_loss=tf.reduce_mean(train_loss+l2/2*(sum([tf.reduce_sum(x**2) for x in self.ug_weight_x])+sum([tf.reduce_sum(x**2) for x in self.ug_weight_h])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_x])+sum([tf.reduce_sum(x**2) for x in self.rg_weight_h])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_x])+sum([tf.reduce_sum(x**2) for x in self.cltm_weight_h])+sum([tf.reduce_sum(x**2) for x in self.weight_o])))
            if self.optimizer=='Gradient':
                opt=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='RMSprop':
                opt=tf.train.RMSPropOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='Momentum':
                opt=tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.99).minimize(train_loss)
            if self.optimizer=='Adam':
                opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)
            with tf.name_scope('train_accuracy'):
                if self.pattern=='1n':
                    train_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,2),tf.argmax(self.labels,2)),tf.float32))
                elif self.pattern=='n1' or self.predicate==True:
                    if self.pattern=='n1':
                        equal=tf.equal(tf.argmax(self.output[-1],1),tf.argmax(self.labels,1))
                        train_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
                    else:
                        train_accuracy=tf.reduce_mean(tf.abs(self.output[-1]-self.labels))
                elif self.pattern=='nn':
                    train_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,2),tf.argmax(self.labels,2)),tf.float32))
            if train_summary_path!=None:
                train_loss_scalar=tf.summary.scalar('train_loss',train_loss)
                train_merging=tf.summary.merge([train_loss_scalar])
                train_accuracy_scalar=tf.summary.scalar('train_accuracy',train_accuracy)
                train_merging=tf.summary.merge([train_accuracy_scalar])
                train_writer=tf.summary.FileWriter(train_summary_path)
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            sess=tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            self.sess=sess
            if self.total_epoch==0:
                epoch=epoch+1
            for i in range(epoch):
                t1=time.time()
                if batch!=None:
                    batches=int((self.shape0-self.shape0%batch)/batch)
                    total_loss=0
                    total_acc=0
                    np.random.shuffle(random)
                    for j in range(batches):
                        index1=j*batch
                        index2=(j+1)*batch
                        if batch!=1:
                            data_batch=self.train_data[index1:index2][random]
                            labels_batch=self.train_labels[index1:index2][random]
                        else:
                            data_batch=self.train_data[random][j]
                            labels_batch=self.train_labels[random][j]
                        feed_dict={self.data:data_batch,self.labels:labels_batch}
                        if i==0 and self.total_epoch==0:
                            batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                        else:
                            batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                        total_loss+=batch_loss
                        batch_acc=sess.run(train_accuracy,feed_dict=feed_dict)
                        total_acc+=batch_acc
                    if self.shape0%batch!=0:
                        batches+=1
                        index1=batches*batch
                        index2=batch-(self.shape0-batches*batch)
                        data_batch=np.concatenate((self.train_data[index1:],self.train_data[:index2]))[random]
                        labels_batch=np.concatenate((self.train_labels[index1:],self.train_labels[:index2]))[random]
                        feed_dict={self.data:data_batch,self.labels:labels_batch}
                        if i==0 and self.total_epoch==0:
                            batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                        else:
                            batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                        total_loss+=batch_loss
                        batch_acc=sess.run(train_accuracy,feed_dict=feed_dict)
                        total_acc+=batch_acc
                    loss=total_loss/batches
                    train_acc=total_acc/batches
                    self.train_loss_list.append(loss.astype(np.float32))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float32)
                    self.train_accuracy_list.append(train_acc.astype(np.float32))
                    self.train_accuracy=train_acc
                    self.train_accuracy=self.train_accuracy.astype(np.float32)
                    if test==True:
                        self.test_loss,self.test_accuracy=self.test(self.test_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_accuracy_list.append(self.test_acc)
                else:
                    feed_dict={self.data:self.train_data,self.labels:self.train_labels}
                    if i==0 and self.total_epoch==0:
                        loss=sess.run(train_loss,feed_dict=feed_dict)
                    else:
                        loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                    self.train_loss_list.append(loss.astype(np.float32))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float32)
                    accuracy=sess.run(train_accuracy,feed_dict=feed_dict)
                    self.train_accuracy_list.append(accuracy.astype(np.float32))
                    self.train_accuracy=accuracy
                    self.train_accuracy=self.train_accuracy.astype(np.float32)
                    if test==True:
                        self.test_loss,self.test_accuracy=self.test(self.test_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_accuracy_list.append(self.test_acc)
                self.epoch+=1
                if epoch%10!=0:
                    d=epoch-epoch%10
                    d=int(d/10)
                else:
                    d=epoch/10
                if d==0:
                    d=1
                if i%d==0:
                    if continue_train==True:
                        print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                    else:
                        print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                    if model_path!=None and i%epoch*2==0:
                        self.save_model(model_path,i,one)
                    if train_summary_path!=None:
                        train_summary=sess.run(train_merging,feed_dict=feed_dict)
                        train_writer.add_summary(train_summary,i)
                t2=time.time()
                self.time+=(t2-t1)
            if model_path!=None:
                self.save_model(model_path)
            self.time=self.time-int(self.time)
            if self.time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            print()
            print('last loss:{0:.6f}'.format(self.train_loss))
            if self.predicate==False:
                print('accuracy:{0:.1f}%'.format(self.train_accuracy*100))
            else:
                print('accuracy:{0:.6f}'.format(self.train_accuracy))
            if train_summary_path!=None:
                train_writer.close()
            if continue_train==True:
                self.last_embedding_w=sess.run(self.embedding_w)
                self.last_embedding_b=sess.run(self.embedding_b)
                self.last_ug_weight_x=sess.run(self.ug_weight_x)
                self.last_ug_weight_h=sess.run(self.ug_weight_h)
                self.last_rg_weight_x=sess.run(self.rg_weight_x)
                self.last_rg_weight_h=sess.run(self.rg_weight_h)
                self.last_cltm_weight_x=sess.run(self.cltm_weight_x)
                self.last_cltm_weight_h=sess.run(self.cltm_weight_h)
                self.last_ug_bias=sess.run(self.ug_bias)
                self.last_rg_bias=sess.run(self.rg_bias)
                self.last_cltm_bias=sess.run(self.cltm_bias)
                self.last_weight_o=sess.run(self.weight_o)
                self.last_bias_o=sess.run(self.bias_o)
                if self.layers!=None:
                    self.last_ug_weight_x=[]
                    self.last_ug_weight_h=[]
                    self.last_rg_weight_x=[]
                    self.last_rg_weight_h=[]
                    self.last_cltm_weight_x=[]
                    self.last_cltm_weight_h=[]
                    self.last_ug_bias=[]
                    self.last_rg_bias=[]
                    self.last_cltm_bias=[]
                    for i in range(self.layers):
                        self.ug_weight_x.append(tf.Variable(self.last_ug_weight_x[i],name='ug_weight_x{}'.format(i+1)))
                        self.ug_weight_h.append(tf.Variable(self.last_ug_weight_h[i],name='ug_weight_h{}'.format(i+1)))
                        self.rg_weight_x.append(tf.Variable(self.last_rg_weight_x[i],name='rg_weight_x{}'.format(i+1)))
                        self.rg_weight_h.append(tf.Variable(self.last_rg_weight_h[i],name='rg_weight_h{}'.format(i+1)))
                        self.cltm_weight_x.append(tf.Variable(self.last_cltm_weight_x[i],name='cltm_weight_x{}'.format(i+1)))
                        self.cltm_weight_h.append(tf.Variable(self.last_cltm_weight_h[i],name='cltm_weight_h{}'.format(i+1)))
                        self.ug_bias.append(tf.Variable(self.last_fg_bias[i],name='ug_bias{}'.format(i+1)))
                        self.rg_bias.append(tf.Variable(self.last_ig_bias[i],name='rg_bias{}'.format(i+1)))
                        self.cltm_bias.append(tf.Variable(self.last_cltm_bias[i],name='cltm_bias{}'.format(i+1)))
                    self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                    self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                else:
                    self.ug_weight_x=tf.Variable(self.last_ug_weight_x,name='ug_weight_x')
                    self.ug_weight_h=tf.Variable(self.last_ug_weight_h,name='ug_weight_h')
                    self.rg_weight_x=tf.Variable(self.last_rg_weight_x,name='rg_weight_x')
                    self.rg_weight_h=tf.Variable(self.last_rg_weight_h,name='rg_weight_h')
                    self.cltm_weight_x=tf.Variable(self.last_cltm_weight_x,name='cltm_weight_x')
                    self.cltm_weight_h=tf.Variable(self.last_cltm_weight_h,name='cltm_weight_h')
                    self.ug_bias=tf.Variable(self.last_ug_bias,name='ug_bias')
                    self.rg_bias=tf.Variable(self.last_rg_bias,name='rg_bias')
                    self.cltm_bias=tf.Variable(self.last_cltm_bias,name='cltm_bias')
                    self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                    self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                self.last_embedding_w=None
                self.last_embedding_b=None
                self.last_ug_weight_x=None
                self.last_ug_weight_h=None
                self.last_rg_weight_x=None
                self.last_rg_weight_h=None
                self.last_cltm_weight_x=None
                self.last_cltm_weight_h=None
                self.last_ug_bias=None
                self.last_rg_bias=None
                self.last_cltm_bias=None
                self.last_weight_o=None
                self.last_bias_o=None
                sess.run(tf.global_variables_initializer())
            print('time:{0}s'.format(self.time))
            return
            
            
    def end(self):
        with self.graph.as_default():
            self.end_flag=True
            self.continue_train=False
            self.last_embedding_w=self.sess.run(self.embedding_w)
            self.last_embedding_b=self.sess.run(self.embedding_b)
            self.last_ug_weight_x=self.sess.run(self.ug_weight_x)
            self.last_ug_weight_h=self.sess.run(self.ug_weight_h)
            self.last_rg_weight_x=self.sess.run(self.rg_weight_x)
            self.last_rg_weight_h=self.sess.run(self.rg_weight_h)
            self.last_cltm_weight_x=self.sess.run(self.cltm_weight_x)
            self.last_cltm_weight_h=self.sess.run(self.cltm_weight_h)
            self.last_ug_bias=self.sess.run(self.ug_bias)
            self.last_rg_bias=self.sess.run(self.rg_bias)
            self.last_cltm_bias=self.sess.run(self.cltm_bias)
            self.last_weight_o=self.sess.run(self.weight_o)
            self.last_bias_o=self.sess.run(self.bias_o)
            self.ug_weight_x=None
            self.ug_weight_h=None
            self.rg_weight_x=None
            self.rg_weight_h=None
            self.cltm_weight_x=None
            self.cltm_weight_h=None
            self.ug_bias=None
            self.rg_bias=None
            self.cltm_bias=None
            self.weight_o=None
            self.bias_o=None
            self.total_epoch+=self.epoch
            self.total_time+=self.time
            self.sess.close()
            return    
    
    
    def test(self,test_data,test_labels,batch=None):
        with self.graph.as_default():
            if len(self.last_weight)==0 or self.test_flag==False:
                use_nn=False
            elif len(self.last_weight)!=0 and self.test_flag!=False:
                use_nn=True
            shape=test_labels.shape
            test_data_placeholder=tf.placeholder(dtype=test_data.dtype,shape=[None,test_data.shape[1],test_data.shape[2]])
            if len(shape)==3:
                test_labels_placeholder=tf.placeholder(dtype=test_labels.dtype,shape=[None,None,shape[2]])
            elif len(shape)==2:
                test_labels_placeholder=tf.placeholder(dtype=test_labels.dtype,shape=[None,shape[1]])
            self.forward_propagation(test_data,use_nn=use_nn)
            if self.pattern=='1n':
                test_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=test_labels_placeholder,axis=2),axis=1))
            elif self.pattern=='n1' or self.predicate==True:
                if self.pattern=='n1':
                    test_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output[-1],labels=test_labels_placeholder))
                else:
                    test_loss=tf.reduce_mean(tf.square(self.output[-1]-tf.expand_dims(test_labels_placeholder,axis=1)))
            elif self.pattern=='nn':
                test_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=test_labels_placeholder,axis=2),axis=1))
            if self.pattern=='1n':
              test_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,2),tf.argmax(test_labels_placeholder,2)),tf.float32))
            elif self.pattern=='n1' or self.predicate==True:
                if self.pattern=='n1':
                    equal=tf.equal(tf.argmax(self.output[-1],1),tf.argmax(test_labels_placeholder,1))
                    test_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
                else:
                    test_accuracy=tf.reduce_mean(tf.abs(self.output[-1]-test_labels_placeholder))
            elif self.pattern=='nn':
                test_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,2),tf.argmax(test_labels_placeholder,2)),tf.float32))
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            sess=tf.Session(config=config)
            if batch!=None:
                total_test_loss=0
                total_test_acc=0
                test_batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
                for j in range(test_batches):
                    test_data_batch=test_data[j*batch:(j+1)*batch]
                    test_labels_batch=test_labels[j*batch:(j+1)*batch]
                    batch_test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_loss+=batch_test_loss
                    batch_test_acc=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_acc+=batch_test_acc
                if test_data.shape[0]%batch!=0:
                    test_batches+=1
                    test_data_batch=np.concatenate([test_data[test_batches*batch:],test_data[:batch-(test_data.shape[0]-test_batches*batch)]])
                    test_labels_batch=np.concatenate([test_labels[test_batches*batch:],test_labels[:batch-(test_labels.shape[0]-test_batches*batch)]])
                    batch_test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_loss+=batch_test_loss
                    batch_test_acc=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_acc+=batch_test_acc
                test_loss=total_test_loss/test_batches
                test_acc=total_test_acc/test_batches
                test_loss=test_loss
                test_accuracy=test_acc
                test_loss=test_loss.astype(np.float32)
                test_accuracy=test_accuracy.astype(np.float32)
            else:
                test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                test_accuracy=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                test_loss=test_loss.astype(np.float32)
                test_accuracy=test_accuracy.astype(np.float32)
            sess.close()
            return test_loss,test_acc
        
        
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.total_epoch))
        print()
        print('l2:{0}'.format(self.l2))
        print()
        print('optimizer:{0}'.format(self.optimizer))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.total_time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0:.6f}'.format(self.train_loss))
        print()
        if self.predicate==False:
            print('train accuracy:{0:.1f}%'.format(self.train_accuracy*100))
        else:
            print('train accuracy:{0:.6f}'.format(self.train_accuracy))
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        print()
        if self.predicate==False:
            print('test accuracy:{0:.1f}%'.format(self.test_accuracy*100))
        else:
            print('test accuracy:{0:.6f}'.format(self.test_accuracy))
        return
		
    
    def info(self):
        self.train_info()
        if self.test_flag==True:
            print()
            print('-------------------------------------')
            self.test_info()
        return


    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.train_accuracy_list)
        plt.title('train accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        print('train loss:{0:.6f}'.format(self.train_loss))
        print()
        if self.predicate==False:
            print('train accuracy:{0:.1f}%'.format(self.train_accuracy*100))
        else:
            print('train accuracy:{0:.6f}'.format(self.train_accuracy))
        return
    
    
    def test_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.test_accuracy_list)
        plt.title('test accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        print('test loss:{0:.6f}'.format(self.test_loss))
        print()
        if self.predicate==False:
            print('test accuracy:{0:.1f}%'.format(self.test_accuracy*100))
        else:
            print('test accuracy:{0:.6f}'.format(self.test_accuracy))
        return
        
    
    def comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch),self.test_loss_list,'r-',label='test loss')
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.train_accuracy_list,'b-',label='train accuracy')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch),self.test_accuracy_list,'r-',label='test accuracy')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        print('train loss:{0:.6f}'.format(self.train_loss))
        print()
        print('train accuracy:{0:.1f}%'.format(self.train_accuracy*100))
        if self.test_flag:
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            print()
            if self.predicate==False:
                print('test accuracy:{0:.1f}%'.format(self.test_accuracy*100))
            else:
                print('test accuracy:{0:.6f}'.format(self.test_accuracy))
        return
    
    
    def network(self):
        print()
        print('input layer\t{0}x{1}'.format(self.data_shape[1],self.data_shape[2]))
        print()
        print('update gate\t{0}\t{1}'.format(self.data_shape[2]*self.hidden+self.hidden*self.hidden+self.hidden,'sigmoid'))
        print()
        print('reset gate\t{0}\t{1}'.format(self.data_shape[2]*self.hidden+self.hidden*self.hidden+self.hidden,'sigmoid'))
        print()
        print('cltm\t{0}\t{1}'.format(self.data_shape[2]*self.hidden+self.hidden*self.hidden+self.hidden,'tanh'))
        print()
        print('hidden layer\t{0}'.format(self.hidden))
        print()
        if self.layers!=None:
            print('layers\t{0}'.format(self.layers))
            print()
        if len(self.labels_shape)==3:
            print('output layer\t{0}\t{1}'.format(self.labels_shape[2],self.hidden*self.labels_shape[2]+self.labels_shape[2]))
        elif len(self.labels_shape)==2:
            print('output layer\t{0}\t{1}'.format(self.labels_shape[1],self.hidden*self.labels_shape[1]+self.labels_shape[1]))
        print()
        if self.layers!=None:
            if len(self.labels_shape)==3:
                total_params=(self.data_shape[2]*self.hidden+self.hidden*self.hidden+self.hidden)*4+self.hidden*self.labels_shape[2]+self.labels_shape[2]
                total_params+=((self.hidden*self.hidden+self.hidden*self.hidden+self.hidden)*4+self.hidden*self.labels_shape[2]+self.labels_shape[2])*self.layers-1
            elif len(self.labels_shape)==2:
                total_params=(self.data_shape[2]*self.hidden+self.hidden*self.hidden+self.hidden)*4+self.hidden*self.labels_shape[1]+self.labels_shape[1]
                total_params+=((self.hidden*self.hidden+self.hidden*self.hidden+self.hidden)*4+self.hidden*self.labels_shape[1]+self.labels_shape[1])*self.layers-1
        if len(self.labels_shape)==3:
            total_params=(self.data_shape[2]*self.hidden+self.hidden*self.hidden+self.hidden)*4+self.hidden*self.labels_shape[2]+self.labels_shape[2]
        elif len(self.labels_shape)==2:
            total_params=(self.data_shape[2]*self.hidden+self.hidden*self.hidden+self.hidden)*4+self.hidden*self.labels_shape[1]+self.labels_shape[1]
        print('total params:{0}'.format(total_params))
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump([self.embedding_w,self.embedding_b,self.last_ug_weight_x,self.last_ug_weight_h,self.last_rg_weight_x,self.last_rg_weight_h,self.last_cltm_weight_x,self.last_cltm_weight_h,self.last_weight_o,self.last_ug_bias,self.last_rg_bias,self.last_cltm_bias,self.last_bias_o],parameter_file)
        return
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.embedding_w,output_file)
        pickle.dump(self.embedding_b,output_file)
        if self.layers!=None:
            pickle.dump(self.last_ug_weight_x,output_file)
            pickle.dump(self.last_ug_weight_h,output_file)
            pickle.dump(self.last_rg_weight_x,output_file)
            pickle.dump(self.last_rg_weight_h,output_file)
            pickle.dump(self.last_cltm_weight_x,output_file)
            pickle.dump(self.last_cltm_weight_h,output_file)
            pickle.dump(self.last_weight_o,output_file)
            pickle.dump(self.last_ug_bias,output_file)
            pickle.dump(self.last_rg_bias,output_file)
            pickle.dump(self.last_cltm_bias,output_file)
            pickle.dump(self.last_bias_o,output_file) 
        else:
            pickle.dump(self.last_ug_weight_x,output_file)
            pickle.dump(self.last_ug_weight_h,output_file)
            pickle.dump(self.last_rg_weight_x,output_file)
            pickle.dump(self.last_rg_weight_h,output_file)
            pickle.dump(self.last_cltm_weight_x,output_file)
            pickle.dump(self.last_cltm_weight_h,output_file)
            pickle.dump(self.last_weight_o,output_file)
            pickle.dump(self.last_ug_bias,output_file)
            pickle.dump(self.last_rg_bias,output_file)
            pickle.dump(self.last_cltm_bias,output_file)
            pickle.dump(self.last_bias_o,output_file)
        pickle.dump(self.data_dtype,output_file)
        pickle.dump(self.labels_dtype,output_file)
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.data_shape,output_file)
        pickle.dump(self.labels_shape,output_file)
        pickle.dump(self.hidden,output_file)
        pickle.dump(self.pattern,output_file)
        pickle.dump(self.predicate,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.l2,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_accuracy,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_accuracy_list,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_accuracy,output_file)
            pickle.dump(self.test_loss_list,output_file)
            pickle.dump(self.test_accuracy_list,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        input_file=open(model_path,'rb')
        self.last_embedding_w=pickle.load(input_file)
        self.last_embedding_b=pickle.load(input_file)
        if self.layers!=None:
            self.last_ug_weight_x=pickle.load(input_file)
            self.last_ug_weight_h=pickle.load(input_file)
            self.last_rg_weight_x=pickle.load(input_file)
            self.last_rg_weight_h=pickle.load(input_file)
            self.last_cltm_weight_x=pickle.load(input_file)
            self.last_cltm_weight_h=pickle.load(input_file)
            self.last_weight_o=pickle.load(input_file)
            self.last_ug_bias=pickle.load(input_file)
            self.last_rg_bias=pickle.load(input_file)
            self.last_cltm_bias=pickle.load(input_file)
            self.last_bias_o=pickle.load(input_file)
        else:
            self.last_ug_weight_x=pickle.load(input_file)
            self.last_ug_weight_h=pickle.load(input_file)
            self.last_rg_weight_x=pickle.load(input_file)
            self.last_rg_weight_h=pickle.load(input_file)
            self.last_cltm_weight_x=pickle.load(input_file)
            self.last_cltm_weight_h=pickle.load(input_file)
            self.last_weight_o=pickle.load(input_file)
            self.last_ug_bias=pickle.load(input_file)
            self.last_rg_bias=pickle.load(input_file)
            self.last_cltm_bias=pickle.load(input_file)
            self.last_bias_o=pickle.load(input_file)
        self.data_dtype=pickle.load(input_file)
        self.labels_dtype=pickle.load(input_file)
        self.shape0=pickle.load(input_file)
        self.data_shape=pickle.load(input_file)
        self.labels_shape=pickle.load(input_file)
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.data=tf.placeholder(dtype=self.data_dtype,shape=[None,None,None],name='data')
            if len(self.labels_shape)==3:
                self.labels=tf.placeholder(dtype=self.labels_dtype,shape=[None,None,None],name='labels')
            elif len(self.labels_shape)==2:
                self.labels=tf.placeholder(dtype=self.labels_dtype,shape=[None,None],name='labels')
        self.hidden=pickle.load(input_file)
        self.pattern=pickle.load(input_file)
        self.predicate=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.l2=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_accuracy=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_accuracy_list=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_accuracy=pickle.load(input_file)
            self.test_loss_list=pickle.load(input_file)
            self.test_accuracy_list=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        self.flag=1
        input_file.close()
        return


    def classify(self,data,one_hot=False,save_path=None,processor=None):
        with self.graph.as_default():
            if processor!=None:
                self.processor=processor
            self.C.clear()
            self.h.clear()
            data=tf.constant(data)
            self.forward_propagation(data,use_nn=True)
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            with tf.Session(config=config) as sess:
                if self.pattern=='1n':
                    _output=sess.run(self.output)
                elif self.pattern=='n1':
                    _output=sess.run(self.output[-1])
                elif self.pattern=='nn':
                    _output=sess.run(self.output)
                if one_hot==True:
                    if len(_output.shape)==2:
                        index=np.argmax(_output,axis=1)
                        output=np.zeros([_output.shape[0],_output.shape[1]])
                        for i in range(_output.shape[0]):
                            output[i][index[i]]+=1
                    else:
                        output=np.zeros([_output.shape[0],_output.shape[1],_output[2]])
                        for i in range(_output.shape[0]):
                            index=np.argmax(_output[i],axis=1)
                            for j in range(index.shape[0]):
                                output[i][j][index[j]]+=1
                    if save_path!=None:
                        output_file=open(save_path,'wb')
                        pickle.dump(output,output_file)
                        output_file.close()
                    return output
                else:
                    if len(_output.shape)==2:
                        output=np.argmax(_output,axis=1)+1
                    else:
                        for i in range(_output.shape[0]):
                            output[i]=np.argmax(_output[i],axis=1)+1
                    if save_path!=None:
                        output_file=open(save_path,'wb')
                        pickle.dump(output,output_file)
                        output_file.close()
                    return output
                    
                    
    def predicate(self,data,save_path=None,processor=None):
        with self.graph.as_default():
            if processor!=None:
                self.processor=processor
            self.C.clear()
            self.h.clear()
            data=tf.constant(data)
            self.forward_propagation(data,use_nn=True)
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            with tf.Session(config=config) as sess:
                output=sess.run(self.output[-1])
            if save_path!=None:
                output_file=open(save_path,'wb')
                pickle.dump(output,output_file)
                output_file.close()
            return output
