import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time


class RNN:
    def __init__(self,train_data=None,train_labels=None,test_data=None,test_labels=None):
        self.graph=tf.Graph()
        self.train_data=train_data
        self.train_labels=train_labels
        self.test_data=test_data
        self.test_labels=test_labels
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
        self.embedding_w=None
        self.embedding_b=None
        self.weight_x=None
        self.weight_h=None
        self.weight_o=None
        self.bias_x=None        
        self.bias_h=None      
        self.bias_o=None
        self.h=[]
        self.o=None
        self.last_embedding_w=None
        self.last_embedding_b=None
        self.last_weight_x=None
        self.last_weight_h=None
        self.last_weight_o=None
        self.last_bias_x=None
        self.last_bias_h=None
        self.last_bias_o=None
        self.batch=None
        self.epoch=0
        self.l2=None
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
        self.ooo=False
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

    
    def structure(self,hidden,pattern,ed=None,predicate=False,mean=0,stddev=0.07,dtype=tf.float32):
        with self.graph.as_default():
            self.continue_train=False
            self.flag=None
            self.end_flag=False
            self.test_flag=False
            self.h.clear()
            self.train_loss_list.clear()
            self.train_accuracy_list.clear()
            self.hidden=hidden
            self.pattern=pattern
            self.predicate=predicate
            self.epoch=0
            self.dtype=dtype
            self.total_epoch=0
            self.time=0
            self.total_time=0
            with tf.name_scope('parameter_initialization'):
                self.weight_x=self.weight_init([self.data_shape[2],self.hidden],mean=mean,stddev=stddev,name='weight_x')
                self.bias_x=self.bias_init([self.hidden],mean=mean,stddev=stddev,name='bias_x')
                self.weight_h=self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='weight_h')
                self.bias_h=self.bias_init([self.hidden],mean=mean,stddev=stddev,name='bias_a')
                if len(self.labels_shape)==3:
                    self.weight_o=self.weight_init([self.hidden,self.labels_shape[2]],mean=mean,stddev=stddev,name='weight_o')
                    self.bias_o=self.bias_init([self.labels_shape[2]],mean=mean,stddev=stddev,name='bias_o')
                elif len(self.labels_shape)==2:
                    self.weight_o=self.weight_init([self.hidden,self.labels_shape[1]],mean=mean,stddev=stddev,name='weight_o')
                    self.bias_o=self.bias_init([self.labels_shape[1]],mean=mean,stddev=stddev,name='bias_o')
                self.h.append(tf.zeros([1,self.hidden],name='h0'))
                return
            
                
    def forward_propagation(self,data,use_nn=False):
        with self.graph.as_default():
            self.o=None
            if use_nn==False:
                embedding_w=self.embedding_w
                embedding_b=self.embedding_b
                weight_x=self.weight_x
                weight_h=self.weight_h
                weight_o=self.weight_o
                bias_x=self.bias_x
                bias_h=self.bias_h
                bias_o=self.bias_o
            else:
                embedding_w=tf.constant(self.last_embedding_w)
                embedding_b=tf.constant(self.last_embedding_b)
                weight_x=tf.constant(self.last_weight_x)
                weight_h=tf.constant(self.last_weight_h)
                weight_o=tf.constant(self.last_weight_o)
                bias_x=tf.constant(self.last_bias_x)
                bias_h=tf.constant(self.last_bias_h)
                bias_o=tf.constant(self.last_bias_o)
            with tf.device(self.processor):
                with tf.name_scope('forward_propagation'):
                    data=tf.einsum('ijk,kl->ijl',data,embedding_w)+embedding_b
                    X=tf.einsum('ijk,kl->ijl',data,weight_x)+bias_x
                    if self.pattern=='1n':
                        for i in range(self.labels_shape[1]):
                            if i==0:
                                self.h.append(tf.nn.tanh(tf.matmul(self.h[i],weight_h)+X+bias_h))
                            else:
                                self.h.append(tf.nn.tanh(tf.matmul(self.h[i],weight_h)+bias_h))
                        self.o=tf.einsum('ijk,kl->ijl',tf.stack(self.h[1:],axis=1),weight_o)+bias_o
                    elif self.pattern=='n1' or self.predicate==True:
                        for i in range(self.data_shape[1]):
                            self.h.append(tf.nn.tanh(tf.matmul(self.h[i],weight_h)+X[:][:,i]+bias_h))
                            self.o.append(tf.add(tf.matmul(self.h[i+1],weight_o),bias_o))
                    elif self.pattern=='nn':
                        for i in range(self.data_shape[1]):
                            self.h.append(tf.nn.tanh(tf.matmul(self.h[i],weight_h)+X[:][:,i]+bias_h))
                        self.o=tf.einsum('ijk,kl->ijl',tf.stack(self.h[1:],axis=1),weight_o)+bias_o
                    self.h.remove(self.h[0])
                    return
            
        
    def train(self,batch=None,epoch=None,optimizer='Adam',lr=0.001,l2=None,test=False,test_batch=None,train_summary_path=None,model_path=None,one=True,continue_train=False,processor=None):
        with self.graph.as_default():
            self.h.clear()
            self.h.append(tf.zeros([1,self.hidden],name='h0'))
            self.batch=batch
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
                self.embedding_w=tf.Variable(self.last_embedding_w,name='embedding_w')
                self.embedding_b=tf.Variable(self.last_embedding_b,name='embedding_b')
                self.weight_x=tf.Variable(self.last_weight_x,name='weight_x')
                self.weight_h=tf.Variable(self.last_weight_h,name='weight_h')
                self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                self.bias_x=tf.Variable(self.last_bias_x,name='bias_x')
                self.bias_h=tf.Variable(self.last_bias_h,name='bias_h')
                self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                self.last_embedding_w=None
                self.last_embedding_b=None
                self.last_weight_x=None
                self.last_weight_h=None
                self.last_weight_o=None
                self.last_bias_x=None
                self.last_bias_h=None
                self.last_bias_o=None
            if continue_train==True and self.flag==1:
                self.flag=0
                self.embedding_w=tf.Variable(self.last_embedding_w,name='embedding_w')
                self.embedding_b=tf.Variable(self.last_embedding_b,name='embedding_b')
                self.weight_x=tf.Variable(self.last_weight_x,name='weight_x')
                self.weight_h=tf.Variable(self.last_weight_h,name='weight_h')
                self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                self.bias_x=tf.Variable(self.last_bias_x,name='bias_x')
                self.bias_h=tf.Variable(self.last_bias_h,name='bias_h')
                self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                self.last_embedding_w=None
                self.last_embedding_b=None
                self.last_weight_x=None
                self.last_weight_h=None
                self.last_weight_o=None
                self.last_bias_x=None
                self.last_bias_h=None
                self.last_bias_o=None
#     －－－－－－－－－－－－－－－forward propagation－－－－－－－－－－－－－－－
            self.forward_propagation(self.data)
#     －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
            with tf.name_scope('train_loss'):
                if self.pattern=='1n':
                    if l2==None:
                        train_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o,labels=self.labels,axis=2),axis=1))
                    else:
                        train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o,labels=self.labels,axis=2),axis=1)
                        train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.weight_x**2)+tf.reduce_sum(self.weight_h**2)+tf.reduce_sum(self.weight_o**2)))
                elif self.pattern=='n1' or self.predicate==True:
                    if self.pattern=='n1':
                        if l2==None:
                            train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o[-1],labels=self.labels))
                        else:
                            train_loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o[-1],labels=self.labels)
                            train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.weight_x**2)+tf.reduce_sum(self.weight_h**2)+tf.reduce_sum(self.weight_o**2)))
                    else:
                        if l2==None:
                            train_loss=tf.reduce_mean(tf.square(self.o[-1]-tf.expand_dims(self.labels,axis=1)))
                        else:
                            train_loss=tf.square(self.o[-1]-tf.expand_dims(self.labels,axis=1))
                            train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.weight_x**2)+tf.reduce_sum(self.weight_h**2)+tf.reduce_sum(self.weight_o**2)))
                elif self.pattern=='nn':
                    if l2==None:
                        train_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o,labels=self.labels,axis=2),axis=1))
                    else:
                        train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o,labels=self.labels,axis=2),axis=1)
                        train_loss=tf.reduce_mean(train_loss+l2/2*(tf.reduce_sum(self.weight_x**2)+tf.reduce_sum(self.weight_h**2)+tf.reduce_sum(self.weight_o**2)))
            if self.optimizer=='Gradient':
                opt=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='RMSprop':
                opt=tf.train.RMSPropOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='Momentum':
                opt=tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.99).minimize(train_loss)
            if self.optimizer=='Adam':
                opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)
            train_loss_scalar=tf.summary.scalar('train_loss',train_loss)
            with tf.name_scope('train_accuracy'):
                if self.pattern=='1n':
                    train_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.o,2),tf.argmax(self.labels,2)),tf.float32))
                elif self.pattern=='n1' or self.predicate==True:
                    if self.pattern=='n1':
                        equal=tf.equal(tf.argmax(self.o[-1],1),tf.argmax(self.labels,1))
                        train_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
                    else:
                        train_accuracy=tf.reduce_mean(tf.abs(self.o[-1]-self.labels))
                elif self.pattern=='nn':
                    train_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.o,2),tf.argmax(self.labels,2)),tf.float32))
                train_accuracy_scalar=tf.summary.scalar('train_accuracy',train_accuracy)
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
            random=np.arange(self.shape0)
            for i in range(epoch):
                t1=time.time()
                if batch!=None:
                    batches=int((self.shape0-self.shape0%batch)/batch)
                    total_loss=0
                    total_acc=0
                    np.random.shuffle(random)
                    if self.ooo==True:
                        self.train_data=self.train_data[random]
                        self.train_labels=self.train_labels[random]
                    for j in range(batches):
                        index1=j*batch
                        index2=(j+1)*batch
                        if self.ooo==False:
                            train_data_batch=self.train_data[random[index1:index2]]
                            train_labels_batch=self.train_labels[random[index1:index2]]
                        else:
                            train_data_batch=self.train_data[index1:index2]
                            train_labels_batch=self.train_labels[index1:index2]
                        feed_dict={self.data:train_data_batch,self.labels:train_labels_batch}
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
                        if self.ooo==False:
                            train_data_batch=self.train_data[np.concatenate([random[index1:],random[:index2]])]
                            train_labels_batch=self.train_data[np.concatenate([random[index1:],random[:index2]])]
                        else:
                            train_data_batch=np.concatenate([self.train_data[index1:],self.train_data[:index2]])
                            train_labels_batch=np.concatenate([self.train_labels[index1:],self.train_labels[:index2]])
                        feed_dict={self.data:train_data_batch,self.labels:train_labels_batch}
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
                    np.random.shuffle(random)
                    if self.ooo==False:
                        train_data=self.train_data[random]
                        train_labels=self.train_labels[random]
                        feed_dict={self.data:train_data,self.labels:train_labels}
                    else:
                        self.train_data=self.train_data[random]
                        self.train_labels=self.train_labels[random]
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
                        self.save(model_path,i,one)
                    if train_summary_path!=None:
                        train_summary=sess.run(train_merging,feed_dict=feed_dict)
                        train_writer.add_summary(train_summary,i)
                t2=time.time()
                self.time+=(t2-t1)
            self.time=self.time-int(self.time)
            if self.time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            print()
            print('last loss:{0:.6f}'.format(self.train_loss))
            print('accuracy:{0:.1f}%'.format(self.train_accuracy*100))
            if train_summary_path!=None:
                train_writer.close()
            if continue_train==True:
                self.last_embedding_w=sess.run(self.embedding_w)
                self.last_embedding_b=sess.run(self.embedding_b)
                self.last_weight_x=sess.run(self.weight_x)
                self.last_weight_h=sess.run(self.weight_h)
                self.last_weight_o=sess.run(self.weight_o)
                self.last_bias_x=sess.run(self.bias_x)
                self.last_bias_h=sess.run(self.bias_h)
                self.last_bias_o=sess.run(self.bias_o)
                self.weight_x=tf.Variable(self.last_weight_x,name='weight_x')
                self.weight_h=tf.Variable(self.last_weight_h,name='weight_h')
                self.weight_o=tf.Variable(self.last_weight_o,name='weight_o')
                self.bias_x=tf.Variable(self.last_bias_x,name='bias_x')
                self.bias_h=tf.Variable(self.last_bias_h,name='bias_h')
                self.bias_o=tf.Variable(self.last_bias_o,name='bias_o')
                self.last_embedding_w=None
                self.last_embedding_b=None
                self.last_weight_x=None
                self.last_weight_h=None
                self.last_weight_o=None
                self.last_bias_x=None
                self.last_bias_h=None
                self.last_bias_o=None
                sess.run(tf.global_variables_initializer())
            if continue_train==True:
                if self.total_epoch==0:
                    self.total_epoch=epoch-1
                    self.epoch=epoch-1
                else:
                    self.total_epoch=self.total_epoch+epoch
                    self.epoch=epoch
            if continue_train!=True:
                self.epoch=epoch-1
            print('time:{0}s'.format(self.time))
            return
        
        
    def end(self):
        with self.graph.as_default():
            self.end_flag=True
            self.continue_train=False
            self.last_embedding_w=self.sess.run(self.embedding_w)
            self.last_embedding_b=self.sess.run(self.embedding_b)
            self.last_weight_x=self.sess.run(self.weight_x)
            self.last_weight_h=self.sess.run(self.weight_h)
            self.last_weight_o=self.sess.run(self.weight_o)
            self.last_bias_x=self.sess.run(self.bias_x)
            self.last_bias_h=self.sess.run(self.bias_h)
            self.last_bias_o=self.sess.run(self.bias_o)
            self.embedding_w=None
            self.embedding_b=None
            self.weight_x=None
            self.weight_h=None
            self.weight_o=None
            self.bias_x=None
            self.bias_h=None
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
            if len(self.labels_shape)==3:
                test_labels_placeholder=tf.placeholder(dtype=test_labels.dtype,shape=[None,None,shape[2]])
            elif len(self.labels_shape)==2:
                test_labels_placeholder=tf.placeholder(dtype=test_labels.dtype,shape=[None,shape[1]])
            self.forward_propagation(test_data_placeholder,use_nn=use_nn)
            if self.pattern=='1n':
                test_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o,labels=test_labels_placeholder,axis=2),axis=1))
            elif self.pattern=='n1' or self.predicate==True:
                if self.pattern=='n1':
                    test_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o[-1],labels=test_labels_placeholder))
                else:
                    test_loss=tf.reduce_mean(tf.square(self.o[-1]-tf.expand_dims(test_labels_placeholder,axis=1)))
            elif self.pattern=='nn':
                test_loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.o,labels=test_labels_placeholder,axis=2),axis=1))
            if self.pattern=='1n':
                test_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.o,2),tf.argmax(test_labels_placeholder,2)),tf.float32))
            elif self.pattern=='n1' or self.predicate==True:
                if self.pattern=='n1':
                    equal=tf.equal(tf.argmax(self.o[-1],1),tf.argmax(test_labels_placeholder,1))
                    test_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
                else:
                    test_accuracy=tf.reduce_mean(tf.abs(self.o[-1]-test_labels_placeholder))
            elif self.pattern=='nn':
                test_accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.o,2),tf.argmax(test_labels_placeholder,2)),tf.float32))
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
                test_accuracy=test_accuracy.astype(np.float16)
            else:
                test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                test_accuracy=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                test_loss=test_loss.astype(np.float32)
                test_accuracy=test_accuracy.astype(np.float16)
            sess.close()
            return test_loss,test_accuracy  
        
    
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
        print('train loss:{0}'.format(self.train_loss))
        print()
        if self.predicate==False:
            print('train accuracy:{0:.1f}%'.format(self.train_accuracy*100))
        else:
            print('train accuracy:{0:.6f}'.format(self.train_accuracy))
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0}'.format(self.test_loss))
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
        print('train loss:{0}'.format(self.train_loss))
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
        print('train loss:{0}'.format(self.train_loss))
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
        print('input layer\t{0}x{1}\t{2}'.format(self.data_shape[1],self.data_shape[2],self.data_shape[2]*self.hidden+self.hidden))
        print()
        print('hidden layer\t{0}\t{1}\t{2}'.format(self.hidden,self.hidden*self.hidden+self.hidden,'tanh'))
        print()
        if len(self.labels_shape)==3:
            print('output layer\t{0}\t{1}'.format(self.labels_shape[2],self.hidden*self.labels_shape[2]+self.labels_shape[2]))
        elif len(self.labels_shape)==2:
            print('output layer\t{0}\t{1}'.format(self.labels_shape[1],self.hidden*self.labels_shape[1]+self.labels_shape[1]))
        print()
        if len(self.labels_shape)==3:
            total_params=self.data_shape[2]*self.hidden+self.hidden+self.hidden*self.hidden+self.hidden+self.hidden*self.labels_shape[2]+self.labels_shape[2]
        elif len(self.labels_shape)==2:
            total_params=self.data_shape[2]*self.hidden+self.hidden+self.hidden*self.hidden+self.hidden+self.hidden*self.labels_shape[1]+self.labels_shape[1]
        print('total params:{0}'.format(total_params))
        return


    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.embedding_w,output_file)
        pickle.dump(self.embedding_b,output_file)
        pickle.dump(self.last_weight_x,output_file)
        pickle.dump(self.last_weight_h,output_file)
        pickle.dump(self.last_weight_o,output_file)
        pickle.dump(self.last_bias_x,output_file)
        pickle.dump(self.last_bias_h,output_file)
        pickle.dump(self.last_bias_o,output_file)
        pickle.dump(self.data_dtype,output_file)
        pickle.dump(self.labels_dtype,output_file)
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.data_shape,output_file)
        pickle.dump(self.labels_shape,output_file)
        pickle.dump(self.hidden,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.l2,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(float(self.train_loss),output_file)
        pickle.dump(float(self.train_accuracy*100),output_file)
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
        self.last_weight_x=pickle.load(input_file)
        self.last_weight_h=pickle.load(input_file)
        self.last_weight_o=pickle.load(input_file)
        self.last_bias_x=pickle.load(input_file)
        self.last_bias_h=pickle.load(input_file)
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


    def classify(self,data,one_hot=False,save_path=None,save_csv=None,processor=None):
        with self.graph.as_default():
            if processor!=None:
                self.processor=processor
            self.h.clear()
            data=tf.constant(data)
            self.forward_propagation(data,use_nn=True)
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            with tf.Session(config=config) as sess:
                if self.pattern=='1n':
                    _output=sess.run(self.o)
                elif self.pattern=='n1':
                    _output=sess.run(self.o[-1])
                elif self.pattern=='nn':
                    _output=sess.run(self.o)
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
                    elif save_csv!=None:
                        data=pd.DataFrame(output)
                        data.to_csv(save_csv,index=False,header=False)
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
                    elif save_csv!=None:
                        data=pd.DataFrame(output)
                        data.to_csv(save_csv,index=False,header=False)
                    return output
                    
                    
    def predicate(self,data,save_path=None,save_csv=None,processor=None):
        with self.graph.as_default():
            if processor!=None:
                self.processor=processor
            self.h.clear()
            data=tf.constant(data)
            self.forward_propagation(data,use_nn=True)
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            with tf.Session(config=config) as sess:
                output=sess.run(self.o[-1])
            if save_path!=None:
                output_file=open(save_path,'wb')
                pickle.dump(output,output_file)
                output_file.close()
            elif save_csv!=None:
                data=pd.DataFrame(output)
                data.to_csv(save_csv,index=False,header=False)
            return output
