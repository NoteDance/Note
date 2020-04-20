import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time


def write_data(data,path):
    output_file=open(path,'wb')
    pickle.dump(data,output_file)
    output_file.close()
    return


def read_data(path,dtype=np.float32):
    input_file=open(path,'rb')
    data=pickle.load(input_file)
    return np.array(data,dtype=dtype)


def write_data_csv(data,path,dtype=None,index=False,header=False):
    if dtype==None:
        data=pd.DataFrame(data)
        data.to_csv(path,index=index,header=header)
    else:
        data=np.array(data,dtype=dtype)
        data=pd.DataFrame(data)
        data.to_csv(path,index=index,header=header)
    return
        

def read_data_csv(path,dtype=None,header=None):
    if dtype==None:
        data=pd.read_csv(path,header=header)
        return np.array(data)
    else:
        data=pd.read_csv(path,header=header)
        return np.array(data,dtype=dtype)


class fnn:
    def __init__(self,train_data=None,train_labels=None):
        self.graph=tf.Graph()
        self.train_data=train_data
        self.train_labels=train_labels
        self.pre_train_data=None
        with self.graph.as_default():
            if type(train_data)==np.ndarray:
                self.data_shape=train_data.shape
                self.labels_shape=train_labels.shape
                self.data=tf.placeholder(dtype=train_data.dtype,shape=[None,self.data_shape[1]],name='data')
                if len(self.labels_shape)==2:
                    self.labels=tf.placeholder(dtype=train_labels.dtype,shape=[None,self.labels_shape[1]],name='labels')
                else:
                    self.labels=tf.placeholder(dtype=train_labels.dtype,shape=[None],name='labels')
                self.data_dtype=train_data.dtype
                self.labels_dtype=train_labels.dtype
            else:
                self.data=tf.placeholder(dtype=self.data_dtype,shape=[None,self.data_shape[1]])
                if len(self.labels_shape)==2:
                    self.labels=tf.placeholder(dtype=self.labels_dtype,shape=[None,self.labels_shape[1]])
                else:
                    self.labels=tf.placeholder(dtype=self.labels_dtype,shape=[None])
        self.hidden=[]
        self.hidden_layers=None
        self.layers=None
        self.function=[]
        self.weight=[]
        self.bias=[]
        self.last_weight=[]
        self.last_bias=[]
        self.activation=[]
        self.batch=None
        self.epoch=None
        self.l2=None
        self.dropout=None
        self.optimizer=None
        self.lr=None
        self.train_loss=None
        self.train_accuracy=None
        self.train_loss_list=[]
        self.train_accuracy_list=[]
        self.test_loss=None
        self.test_accuracy=None
        self.normalize=None
        self.maximun=False
        self.continue_train=False
        self.flag=None
        self.end_flag=False
        self.test_flag=None
        self.time=None
        self.cpu_gpu='/gpu:0'
        self.use_cpu_gpu='/gpu:0'
        
        
    def preprocess(self,dtype=np.float32,normalize=True,maximun=False):
        self.normalize=normalize
        self.maximun=maximun
        if self.normalize==True:
            self.pre_train_data=self.train_data
            if self.maximun==True:
                self.pre_train_data/=np.max(self.pre_train_data,axis=0)
            else:
                self.pre_train_data-=np.mean(self.pre_train_data,axis=0)
                self.pre_train_data/=np.std(self.pre_train_data,axis=0)
        return

    
    def test_preprocess(self,data):
        if self.normalize==True:
            if self.maximun==True:
                data/=np.max(data,axis=0)
            else:
                data-=np.mean(data,axis=0)
                data/=np.std(data,axis=0)
        return
    
    
    def weight_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
            
    
    def bias_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)

    
    def structure(self,hidden,function,layers=None,mean=0,stddev=0.07,dtype=np.float32):
        with self.graph.as_default():
            self.continue_train=False
            self.total_epoch=0
            self.flag=None
            self.end_flag=False
            self.test_flag=False
            self.train_loss_list.clear()
            self.train_accuracy_list.clear()
            self.weight.clear()
            self.bias.clear()
            self.last_weight=[]
            self.last_bias=[]
            self.hidden=hidden
            self.function=function
            self.layers=layers
            self.dtype=dtype
            self.time=None
            with tf.name_scope('parameter_initialization'):
                if self.layers!=None:
                    self.hidden_layers=self.layers-2
                    for i in range(self.hidden_layers+1):
                        if i==0:
                            self.weight.append(self.weight_init([self.data_shape[1],self.hidden],mean=mean,stddev=stddev,name='weight_{0}'.format(i+1)))
                            self.bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='bias_{0}'.format(i+1)))
                        if i==self.hidden_layers:
                            if len(self.labels_shape)==2:
                                self.weight.append(self.weight_init([self.hidden,self.labels_shape[1]],mean=mean,stddev=stddev,name='weight_output'))
                                self.bias.append(self.bias_init([self.labels_shape[1]],mean=mean,stddev=stddev,name='bias_output'))
                            else:
                                self.weight.append(self.weight_init([self.hidden,1],mean=mean,stddev=stddev,name='weight_output'))
                                self.bias.append(self.bias_init([1],mean=mean,stddev=stddev,name='bias_output'))
                        elif i>0 and i<self.hidden_layers:
                            self.weight.append(self.weight_init([self.hidden,self.hidden],mean=mean,stddev=stddev,name='weight_{0}'.format(i+1)))
                            self.bias.append(self.bias_init([self.hidden],mean=mean,stddev=stddev,name='bias_{0}'.format(i+1)))
                else:
                    self.hidden_layers=len(self.hidden)
                    for i in range(len(self.hidden)+1):
                        if i==0:
                            self.weight.append(self.weight_init([self.data_shape[1],self.hidden[i]],mean=mean,stddev=stddev,name='weight_{0}'.format(i+1)))
                            self.bias.append(self.bias_init([self.hidden[i]],mean=mean,stddev=stddev,name='bias_{0}'.format(i+1)))
                        if i==len(self.hidden):
                            if len(self.labels_shape)==2:
                                self.weight.append(self.weight_init([self.hidden[i-1],self.labels_shape[1]],mean=mean,stddev=stddev,name='weight_output'))
                                self.bias.append(self.bias_init([self.labels_shape[1]],mean=mean,stddev=stddev,name='bias_output'))
                            else:
                                self.weight.append(self.weight_init([self.hidden,1],mean=mean,stddev=stddev,name='weight_output'))
                                self.bias.append(self.bias_init([1],mean=mean,stddev=stddev,name='bias_output'))
                        elif i>0 and i<=len(self.hidden)-1:
                            self.weight.append(self.weight_init([self.hidden[i-1],self.hidden[i]],mean=mean,stddev=stddev,name='weight_{0}'.format(i+1)))
                            self.bias.append(self.bias_init([self.hidden[i]],mean=mean,stddev=stddev,name='bias_{0}'.format(i+1)))
                return
       
         
    def forward_propagation(self,data,dropout=None,use_nn=False):
        with self.graph.as_default():
            forward_cpu_gpu=[]
            for i in range(self.hidden_layers):
                if type(self.cpu_gpu)==str:
                    forward_cpu_gpu.append(self.cpu_gpu)
                elif len(self.cpu_gpu)!=self.hidden_layers:
                    forward_cpu_gpu.append(self.cpu_gpu[0])
                else:
                    forward_cpu_gpu.append(self.cpu_gpu[i])
            if use_nn==True:
                for i in range(self.hidden_layers):
                    if type(self.use_cpu_gpu)==str:
                        forward_cpu_gpu.append(self.use_cpu_gpu)
                    else:
                        forward_cpu_gpu.append(self.use_cpu_gpu[i])
            if use_nn==False:
                weight=self.weight
                bias=self.bias
            else:
                weight=[]
                bias=[]
                for i in range(len(self.last_weight)):
                    weight.append(tf.constant(self.last_weight[i]))
                    bias.append(tf.constant(self.last_bias[i]))
            self.activation=[x for x in range(self.hidden_layers)]
            if type(dropout)==list:
                data=tf.nn.dropout(data,dropout[0])
            with tf.name_scope('forward_propagation'):
                for i in range(self.hidden_layers):
                    with tf.device(forward_cpu_gpu[i]):
                        if type(self.function)==list:
                            if self.function[i]=='sigmoid':
                                if i==0:
                                    self.activation[i]=tf.nn.sigmoid(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.sigmoid(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                            if self.function[i]=='tanh':
                                if i==0:
                                    self.activation[i]=tf.nn.tanh(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.tanh(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                            if self.function[i]=='relu':
                                if i==0:
                                    self.activation[i]=tf.nn.relu(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.relu(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                            if self.function[i]=='elu':
                                if i==0:
                                    self.activation[i]=tf.nn.elu(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.elu(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                        elif type(self.function)==str:
                            if self.function=='sigmoid':
                                if i==0:
                                    self.activation[i]=tf.nn.sigmoid(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.sigmoid(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                            if self.function=='tanh':
                                if i==0:
                                    self.activation[i]=tf.nn.tanh(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.tanh(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                            if self.function=='relu':
                                if i==0:
                                    self.activation[i]=tf.nn.relu(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.relu(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                            if self.function=='elu':
                                if i==0:
                                    self.activation[i]=tf.nn.elu(tf.matmul(data,weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                                else:
                                    self.activation[i]=tf.nn.elu(tf.matmul(self.activation[i-1],weight[i])+bias[i])
                                    if type(dropout)==list:
                                        self.activation[i]=tf.nn.dropout(self.activation[i],dropout[i+1])
                if dropout!=None and type(dropout)!=list:
                    self.activation[-1]=tf.nn.dropout(self.activation[-1],dropout)
                output=tf.matmul(self.activation[-1],weight[-1])+bias[-1]
                return output
            
            
    def train(self,batch=None,epoch=None,optimizer='Adam',lr=0.001,l2=None,dropout=None,acc=True,train_summary_path=None,model_path=None,one=True,continue_train=False,cpu_gpu=None):
        t1=time.time()
        with self.graph.as_default():
            self.batch=batch
            self.l2=l2
            self.dropout=dropout
            self.optimizer=optimizer
            self.lr=lr
            if continue_train!=True:
                if self.continue_train==True:
                    continue_train=True
                else:
                    self.train_loss_list.clear()
                    self.train_accuracy_list.clear()
            if self.continue_train==False and continue_train==True:
                if self.end_flag==False and self.flag==0:
                    self.epoch=None
                self.train_loss_list.clear()
                self.train_accuracy_list.clear()
                self.continue_train=True
            if cpu_gpu!=None:
                self.cpu_gpu=cpu_gpu
            if type(self.cpu_gpu)==list and (len(self.cpu_gpu)!=self.hidden_layers+1 or len(self.cpu_gpu)==1):
                self.cpu_gpu.append('/gpu:0')
            if type(self.cpu_gpu)==str:
                train_cpu_gpu=self.cpu_gpu
            else:
                train_cpu_gpu=self.cpu_gpu[-1]
            with tf.device(train_cpu_gpu):
                if continue_train==True and self.end_flag==True:
                    self.end_flag=False
                    self.weight=[x for x in range(self.hidden_layers+1)]
                    self.bias=[x for x in range(self.hidden_layers+1)]
                    for i in range(self.hidden_layers+1):
                        if i==self.hidden_layers:
                            self.weight[i]=tf.Variable(self.last_weight[i],name='weight_output')
                            self.bias[i]=tf.Variable(self.last_bias[i],name='bias_output')
                        else:
                            self.weight[i]=tf.Variable(self.last_weight[i],name='weight_{0}'.format(i+1))
                            self.bias[i]=tf.Variable(self.last_bias[i],name='bias_{0}'.format(i+1))
                if continue_train==True and self.flag==1:
                    self.weight=[x for x in range(self.hidden_layers+1)]
                    self.bias=[x for x in range(self.hidden_layers+1)]
                    for i in range(self.hidden_layers+1):
                        if i==self.hidden_layers:
                            self.weight[i]=tf.Variable(self.last_weight[i],name='weight_output')
                            self.bias[i]=tf.Variable(self.last_bias[i],name='bias_output')
                        else:
                            self.weight[i]=tf.Variable(self.last_weight[i],name='weight_{0}'.format(i+1))
                            self.bias[i]=tf.Variable(self.last_bias[i],name='bias_{0}'.format(i+1))
                    self.flag=0
#     －－－－－－－－－－－－－－－forward propagation－－－－－－－－－－－－－－－
                train_output=self.forward_propagation(self.data,self.dropout)
#     －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
                with tf.name_scope('train_loss'):
                    if len(self.labels_shape)==1:
                        if l2==None:
                            train_loss=tf.reduce_mean(tf.square(train_output-tf.expand_dims(self.labels,axis=1)))
                        else:
                            train_loss=tf.square(train_output-tf.expand_dims(self.labels,axis=1))
                            train_loss=tf.reduce_mean(train_loss+l2/2*sum([tf.reduce_sum(x**2) for x in self.weight]))
                    elif self.labels_shape[1]==1:
                        if l2==None:
                            train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=train_output,labels=self.labels))
                        else:
                            train_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=train_output,labels=self.labels)
                            train_loss=tf.reduce_mean(train_loss+l2/2*sum([tf.reduce_sum(x**2) for x in self.weight]))
                    else:
                        if l2==None:
                            train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output,labels=self.labels))
                        else:
                            train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output,labels=self.labels))
                            train_loss=train_loss+l2*sum([tf.reduce_sum(x**2) for x in self.weight])
                    if self.optimizer=='Gradient':
                        opt=tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(train_loss)
                    if self.optimizer=='RMSprop':
                        opt=tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(train_loss)
                    if self.optimizer=='Momentum':
                        opt=tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.99).minimize(train_loss)
                    if self.optimizer=='Adam':
                        opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(train_loss)
                    train_loss_scalar=tf.summary.scalar('train_loss',train_loss)
                    if acc==True:
                        with tf.name_scope('train_accuracy'):
                            if len(self.labels_shape)==1:
                                train_accuracy=tf.reduce_mean(tf.abs(train_output-self.labels))
                            else:
                                equal=tf.equal(tf.argmax(train_output,1),tf.argmax(self.labels,1))
                                train_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
                            train_accuracy_scalar=tf.summary.scalar('train_accuracy',train_accuracy)
                    if train_summary_path!=None:
                        train_merging=tf.summary.merge([train_loss_scalar,train_accuracy_scalar])
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
                        if self.batch!=None:
                            batches=int((self.data_shape[0]-self.data_shape[0]%self.batch)/self.batch)
                            total_loss=0
                            total_acc=0
                            random=np.arange(self.data_shape[0])
                            np.random.shuffle(random)
                            if self.normalize==True:
                                train_data=self.pre_train_data[random]
                            else:
                                train_data=self.train_data[random]
                            train_labels=self.train_labels[random]
                            for j in range(batches):
                                train_data_batch=train_data[j*self.batch:(j+1)*self.batch]
                                train_labels_batch=train_labels[j*self.batch:(j+1)*self.batch]
                                feed_dict={self.data:train_data_batch,self.labels:train_labels_batch}
                                if i==0 and self.total_epoch==0:
                                    batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                                else:
                                    batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                                total_loss+=batch_loss
                                if acc==True:
                                    batch_acc=sess.run(train_accuracy,feed_dict=feed_dict)
                                    total_acc+=batch_acc
                            if self.data_shape[0]%self.batch!=0:
                                batches+=1
                                train_data_batch=np.concatenate([train_data[batches*self.batch:],train_data[:self.batch-(self.data_shape[0]-batches*self.batch)]])
                                train_labels_batch=np.concatenate([train_labels[batches*self.batch:],train_labels[:self.batch-(self.labels_shape[0]-batches*self.batch)]])
                                feed_dict={self.data:train_data_batch,self.labels:train_labels_batch}
                                if i==0 and self.total_epoch==0:
                                    batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                                else:
                                    batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                                total_loss+=batch_loss
                                if acc==True:
                                    batch_acc=sess.run(train_accuracy,feed_dict=feed_dict)
                                    total_acc+=batch_acc
                            loss=total_loss/batches
                            train_acc=total_acc/batches
                            self.train_loss_list.append(float(loss))
                            self.train_loss=loss
                            self.train_loss=self.train_loss.astype(np.float16)
                            if acc==True:
                                self.train_accuracy_list.append(float(train_acc))
                                self.train_accuracy=train_acc
                                self.train_accuracy=self.train_accuracy.astype(np.float16)
                        else:
                            random=np.arange(self.data_shape[0])
                            np.random.shuffle(random)
                            if self.normalize==True:
                                train_data=self.pre_train_data[random]
                            else:
                                train_data=self.train_data[random]
                            train_labels=self.train_labels[random]
                            feed_dict={self.data:train_data,self.labels:train_labels}
                            if i==0 and self.total_epoch==0:
                                loss=sess.run(train_loss,feed_dict=feed_dict)
                            else:
                                loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                            self.train_loss_list.append(float(loss))
                            self.train_loss=loss
                            self.train_loss=self.train_loss.astype(np.float16)
                            if acc==True:
                                if self.normalize==True:
                                    accuracy=sess.run(train_accuracy,feed_dict=feed_dict)
                                else:
                                    accuracy=sess.run(train_accuracy,feed_dict=feed_dict)
                                self.train_accuracy_list.append(float(accuracy))
                                self.train_accuracy=accuracy
                                self.train_accuracy=self.train_accuracy.astype(np.float16)
                        if epoch%10!=0:
                            temp_epoch=epoch-epoch%10
                            temp_epoch=int(temp_epoch/10)
                        else:
                            temp_epoch=epoch/10
                        if temp_epoch==0:
                            temp_epoch=1
                        if i%temp_epoch==0:
                            if continue_train==True:
                                if self.epoch!=None:
                                    self.total_epoch=self.epoch+i+1
                                else:
                                    self.total_epoch=i
                            if continue_train==True:
                                print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                            if model_path!=None and i%epoch*2==0:
                                self.save(model_path,i,one)
                            if train_summary_path!=None:
                                if self.normalize==True:
                                    train_summary=sess.run(train_merging,feed_dict={self.data:self.pre_train_data,self.labels:self.train_labels})
                                    train_writer.add_summary(train_summary,i)
                                else:
                                    train_summary=sess.run(train_merging,feed_dict={self.data:self.train_data,self.labels:self.train_labels})
                                    train_writer.add_summary(train_summary,i)
                    print()
                    print('last loss:{0}'.format(self.train_loss))
                    if acc==True:
                        if len(self.labels_shape)==2:
                            print('accuracy:{0:.3f}%'.format(self.train_accuracy*100))
                        else:
                            print('accuracy:{0:.3f}'.format(self.train_accuracy))
                    if train_summary_path!=None:
                        train_writer.close()
                    if continue_train==True:
                        self.last_weight=sess.run(self.weight)
                        self.last_bias=sess.run(self.bias)
                        for i in range(self.hidden_layers+1):
                            if i==self.hidden_layers:
                                self.weight[i]=tf.Variable(self.last_weight[i],name='weight_output')
                                self.bias[i]=tf.Variable(self.last_bias[i],name='bias_output')
                            else:
                                self.weight[i]=tf.Variable(self.last_weight[i],name='weight_{0}'.format(i+1))
                                self.bias[i]=tf.Variable(self.last_bias[i],name='bias_{0}'.format(i+1))
                        self.last_weight.clear()
                        self.last_bias.clear()
                        sess.run(tf.global_variables_initializer())
                    if continue_train==True:
                        if self.epoch!=None:
                            self.total_epoch=self.epoch+epoch
                        else:
                            self.total_epoch=epoch-1
                        self.epoch=self.total_epoch
                    if continue_train!=True:
                        self.epoch=epoch-1
                    t2=time.time()
                    _time=t2-t1
                    if continue_train!=True or self.time==None:
                        self.time=_time
                    else:
                        self.time+=_time
                    print('time:{0:.3f}s'.format(self.time))
                    return
    
    
    def end(self):
        with self.graph.as_default():
            self.end_flag=True
            self.last_weight=self.sess.run(self.weight)
            self.last_bias=self.sess.run(self.bias)
            self.weight.clear()
            self.bias.clear()
            self.total_epoch=self.epoch
            self.sess.close()
            return
    
    
    def test(self,test_data,test_labels,batch=None):
        with self.graph.as_default():
            if len(self.last_weight)==0 or self.test_flag==False:
                use_nn=False
            elif len(self.last_weight)!=0 and self.test_flag!=False:
                use_nn=True
            self.test_flag=True
            if self.normalize==True:
                test_data=self.test_preprocess(test_data)
            shape=test_labels.shape
            test_data_placeholder=tf.placeholder(dtype=test_data.dtype,shape=[None,test_data.shape[1]])
            if len(shape)==2:
                test_labels_placeholder=tf.placeholder(dtype=test_labels.dtype,shape=[None,shape[1]])
            else:
                test_labels_placeholder=tf.placeholder(dtype=test_labels.dtype,shape=[None])
            test_output=self.forward_propagation(test_data_placeholder,use_nn=use_nn)
            if len(shape)==1:
                test_loss=tf.reduce_mean(tf.square(test_output-test_labels_placeholder))
            elif shape[1]==1:
                test_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_output,labels=test_labels_placeholder))
            else:
                test_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=test_output,labels=test_labels_placeholder))
            if len(shape)==1:
                test_accuracy=tf.reduce_mean(tf.abs(test_output-test_labels_placeholder))
            else:
                equal=tf.equal(tf.argmax(test_output,1),tf.argmax(test_labels_placeholder,1))
                test_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            sess=tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
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
                self.test_loss=test_loss
                self.test_accuracy=test_acc
                self.test_loss=self.test_loss.astype(np.float16)
                self.test_accuracy=self.test_accuracy.astype(np.float16)
            else:
                self.test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                self.test_accuracy=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                self.test_loss=self.test_loss.astype(np.float16)
                self.test_accuracy=self.test_accuracy.astype(np.float16)
            print('test loss:{0}'.format(self.test_loss))
            print('test accuracy:{0:.3f}%'.format(self.test_accuracy*100))
            sess.close()
            return
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.epoch))
        print()
        print('l2:{0}'.format(self.l2))
        print()
        print('dropout:{0}'.format(self.dropout))
        print()
        print('optimizer:{0}'.format(self.optimizer))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0}'.format(self.train_loss))
        print()
        print('train accuracy:{0:.3f}%'.format(self.train_accuracy*100))
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0}'.format(self.test_loss))
        print()
        print('test accuracy:{0:.3f}%'.format(self.test_accuracy*100))
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
        plt.plot(np.arange(self.epoch+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.train_accuracy_list)
        plt.title('train accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        print('train loss:{0}'.format(self.train_loss))
        print()
        print('train accuracy:{0:.3f}%'.format(self.train_accuracy*100))
        return
    
        
    def comparison(self):
        print()
        print('train loss:{0}'.format(self.train_loss))
        print()
        print('train accuracy:{0:.3f}%'.format(self.train_accuracy*100))
        print()
        print('-------------------------------------')
        print()
        print('test loss:{0}'.format(self.test_loss))
        print()
        print('test accuracy:{0:.3f}%'.format(self.test_accuracy*100))
        return
    
    
    def network(self):
        print()
        total_params=0
        if type(self.hidden)==list:
            for i in range(len(self.hidden)+2):
                if i==0:
                    print('input layer\t{0}'.format(self.data_shape[1]))
                    print()
                if i==len(self.hidden)+1:
                    if self.labels_shape[1]==1:
                        print('output layer\t{0}\t{1}\t{2}'.format(self.labels_shape[1],self.labels_shape[1]*self.hidden[i-2]+1,'sigmoid'))
                        total_params+=self.labels_shape[1]*self.hidden[i-2]+1
                        print()
                    else:
                        print('output layer\t{0}\t{1}\t{2}'.format(self.labels_shape[1],self.labels_shape[1]*self.hidden[i-2]+self.labels_shape[1],'softmax'))
                        total_params+=self.labels_shape[1]*self.hidden[i-2]+self.labels_shape[1]
                        print()
                if i>0 and i<len(self.hidden)+1:
                    if i==1:
                        if type(self.function)==list:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden[i-1],self.hidden[i-1]*self.data_shape[1]+self.hidden[i-2],self.function[i-1]))
                            total_params+=self.hidden[i-1]*self.data_shape[1]+self.hidden[i-2]
                            print()
                        else:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden[i-1],self.hidden[i-1]*self.data_shape[1]+self.hidden[i-2],self.function))
                            total_params+=self.hidden[i-1]*self.data_shape[1]+self.hidden[i-2]
                            print()
                    else:
                        if type(self.function)==list:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden[i-1],self.hidden[i-1]*self.hidden[i-1]+self.hidden[i-2],self.function[i-1]))
                            total_params+=self.hidden[i-1]*self.hidden[i-1]+self.hidden[i-2]
                            print()
                        else:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden[i-1],self.hidden[i-1]*self.hidden[i-1]+self.hidden[i-2],self.function))
                            total_params+=self.hidden[i-1]*self.hidden[i-1]+self.hidden[i-2]
                            print()
            print('total params:{0}'.format(total_params))
            return
        else:
            for i in range(self.layers):
                if i==0:
                    print('input layer\t{0}'.format(self.data_shape[1]))
                    print()
                if i==self.layers-1:
                    if self.labels_shape[1]==1:
                        print('output layer\t{0}\t{1}\t{2}'.format(self.labels_shape[1],self.labels_shape[1]*self.hidden+1,'sigmoid'))
                        total_params+=self.labels_shape[1]*self.hidden+1
                        print()
                    else:
                        print('output layer\t{0}\t{1}\t{2}'.format(self.labels_shape[1],self.labels_shape[1]*self.hidden+self.labels_shape[1],'softmax'))
                        total_params+=self.labels_shape[1]*self.hidden+self.labels_shape[1]
                        print()
                if i>0 and i<self.layers-1:
                    if i==1:
                        if type(self.function)==list:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden,self.hidden*self.data_shape[1]+self.hidden,self.function[i-1]))
                            total_params+=self.hidden*self.data_shape[1]+self.hidden
                            print()
                        else:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden,self.hidden*self.data_shape[1]+self.hidden,self.function))
                            total_params+=self.hidden*self.data_shape[1]+self.hidden
                            print()
                    else:
                        if type(self.function)==list:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden,self.hidden*self.hidden+self.hidden,self.function[i-1]))
                            total_params+=self.hidden*self.hidden+self.hidden
                            print()
                        else:
                            print('hidden_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.hidden,self.hidden*self.hidden+self.hidden,self.function))
                            total_params+=self.hidden*self.hidden+self.hidden
                            print()
            print('total params:{0}'.format(total_params))
            return


    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.last_weight,output_file)
        pickle.dump(self.last_bias,output_file)
        pickle.dump(self.data_dtype,output_file)
        pickle.dump(self.labels_dtype,output_file)
        pickle.dump(self.data_shape,output_file)
        pickle.dump(self.labels_shape,output_file)
        pickle.dump(self.hidden_layers,output_file)
        pickle.dump(self.function,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.l2,output_file)
        pickle.dump(self.dropout,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_accuracy,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_accuracy,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_accuracy_list,output_file)
        pickle.dump(self.normalize,output_file)
        pickle.dump(self.maximun,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.time,output_file)
        pickle.dump(self.cpu_gpu,output_file)
        pickle.dump(self.use_cpu_gpu,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        input_file=open(model_path,'rb')
        self.last_weight=pickle.load(input_file)
        self.last_bias=pickle.load(input_file)
        self.data_dtype=pickle.load(input_file)
        self.labels_dtype=pickle.load(input_file)
        self.data_shape=pickle.load(input_file)
        self.labels_shape=pickle.load(input_file)
        self.hidden_layers=pickle.load(input_file)
        self.function=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.l2=pickle.load(input_file)
        self.dropout=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_accuracy=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_accuracy=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_accuracy_list=pickle.load(input_file)
        self.normalize=pickle.load(input_file)
        self.maximun=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        self.cpu_gpu=pickle.load(input_file)
        self.use_cpu_gpu=pickle.load(input_file)
        self.flag=1
        input_file.close()
        return


    def classify(self,data,one_hot=False,save_path=None,save_csv=None,cpu_gpu=None):
        with self.graph.as_default():
            if cpu_gpu!=None:
                self.use_cpu_gpu=cpu_gpu
                use_cpu_gpu=self.use_cpu_gpu[-1]
            with tf.device(use_cpu_gpu):
                if self.normalize==True:
                    if self.maximun==True:
                        data/=np.max(data,axis=0)
                    else:
                        data-=np.mean(data,axis=0)
                        data/=np.std(data,axis=0)
                data=tf.constant(data)
                output=self.forward_propagation(data,use_nn=True)
                config=tf.ConfigProto()
                config.gpu_options.allow_growth=True
                config.allow_soft_placement=True
                with tf.Session(config=config) as sess:
                    output=sess.run(output)
                    if one_hot==True:
                        softmax=np.sum(np.exp(output),axis=1).reshape(output.shape[0],1)
                        softmax=np.exp(output)/softmax
                        index=np.argmax(softmax,axis=1)
                        output=np.zeros([output.shape[0],output.shape[1]])
                        for i in range(output.shape[0]):
                            output[i][index[i]]+=1
                        if save_path!=None:
                            output_file=open(save_path,'wb')
                            pickle.dump(output,output_file)
                            output_file.close()
                        elif save_csv!=None:
                            data=pd.DataFrame(output)
                            data.to_csv(save_csv,index=False,header=False)
                        return output
                    else:
                        softmax=np.sum(np.exp(output),axis=1).reshape(output.shape[0],1)
                        softmax=np.exp(output)/softmax
                        output=np.argmax(softmax,axis=1)+1
                        if save_path!=None:
                            output_file=open(save_path,'wb')
                            pickle.dump(output,output_file)
                            output_file.close()
                        elif save_csv!=None:
                            data=pd.DataFrame(output)
                            data.to_csv(save_csv,index=False,header=False)
                        return output
    
    
    def predicate(self,data,save_path=None,save_csv=None,cpu_gpu=None):
        with self.graph.as_default():
            if cpu_gpu!=None:
                self.use_cpu_gpu=cpu_gpu
                use_cpu_gpu=self.use_cpu_gpu[-1]
            with tf.device(use_cpu_gpu):
                if self.normalize==True:
                    if self.maximun==True:
                        data/=np.max(data,axis=0)
                    else:
                        data-=np.mean(data,axis=0)
                        data/=np.std(data,axis=0)
                    data=tf.constant(data)
                    output=self.forward_propagation(data,use_nn=True)*np.max(self.train_labels)
                else:
                    data=tf.constant(data)
                    output=self.forward_propagation(data,use_nn=True)
                config=tf.ConfigProto()
                config.gpu_options.allow_growth=True
                config.allow_soft_placement=True
                with tf.Session(config=config) as sess:
                    output=sess.run(output)
                if save_path!=None:
                    output_file=open(save_path,'wb')
                    pickle.dump(output,output_file)
                    output_file.close()
                elif save_csv!=None:
                    data=pd.DataFrame(output)
                    data.to_csv(save_csv,index=False,header=False)
                return output
