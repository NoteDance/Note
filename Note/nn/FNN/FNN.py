import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class FNN:
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
                self.data=tf.placeholder(dtype=train_data.dtype,shape=[None,None],name='data')
                if len(self.labels_shape)==2:
                    self.labels=tf.placeholder(dtype=train_labels.dtype,shape=[None,None],name='labels')
                else:
                    self.labels=tf.placeholder(dtype=train_labels.dtype,shape=[None],name='labels')
                self.data_dtype=train_data.dtype
                self.labels_dtype=train_labels.dtype
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
        self.epoch=0
        self.l2=None
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
    
    
    def weight_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
            
    
    def bias_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)

    
    def structure(self,hidden,function,layers=None,mean=0,stddev=0.07,dtype=np.float32):
        with self.graph.as_default():
            self.continue_train=False
            self.flag=None
            self.end_flag=False
            self.test_flag=False
            self.train_loss_list.clear()
            self.train_accuracy_list.clear()
            self.test_loss_list.clear()
            self.test_accuracy_list.clear()
            self.weight.clear()
            self.bias.clear()
            self.last_weight=[]
            self.last_bias=[]
            self.hidden=hidden
            self.function=function
            self.layers=layers
            self.epoch=0
            self.dtype=dtype
            self.total_epoch=0
            self.time=0
            self.total_time=0
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
            processor=[]
            for i in range(self.hidden_layers):
                if type(self.processor)==str:
                    processor.append(self.processor)
                else:
                    processor.append(self.processor[i])
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
                    with tf.device(processor[i]):
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
            
            
    def train(self,batch=None,epoch=None,optimizer='Adam',lr=0.001,l2=None,dropout=None,test=False,test_batch=None,train_summary_path=None,model_path=None,one=True,continue_train=False,processor=None):
        with self.graph.as_default():
            self.batch=batch
            self.epoch=0
            self.l2=l2
            self.dropout=dropout
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
                self.weight=[x for x in range(self.hidden_layers+1)]
                self.bias=[x for x in range(self.hidden_layers+1)]
                for i in range(self.hidden_layers+1):
                    if i==self.hidden_layers:
                        self.weight[i]=tf.Variable(self.last_weight[i],name='weight_output')
                        self.bias[i]=tf.Variable(self.last_bias[i],name='bias_output')
                    else:
                        self.weight[i]=tf.Variable(self.last_weight[i],name='weight_{0}'.format(i+1))
                        self.bias[i]=tf.Variable(self.last_bias[i],name='bias_{0}'.format(i+1))
                self.last_weight.clear()
                self.last_bias.clear()
            if continue_train==True and self.flag==1:
                self.flag=0
                self.weight=[x for x in range(self.hidden_layers+1)]
                self.bias=[x for x in range(self.hidden_layers+1)]
                for i in range(self.hidden_layers+1):
                    if i==self.hidden_layers:
                        self.weight[i]=tf.Variable(self.last_weight[i],name='weight_output')
                        self.bias[i]=tf.Variable(self.last_bias[i],name='bias_output')
                    else:
                        self.weight[i]=tf.Variable(self.last_weight[i],name='weight_{0}'.format(i+1))
                        self.bias[i]=tf.Variable(self.last_bias[i],name='bias_{0}'.format(i+1))
                self.last_weight.clear()
                self.last_bias.clear()
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
                opt=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='RMSprop':
                opt=tf.train.RMSPropOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='Momentum':
                opt=tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.99).minimize(train_loss)
            if self.optimizer=='Adam':
                opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)
            with tf.name_scope('train_accuracy'):
                if len(self.labels_shape)==1:
                    train_accuracy=tf.reduce_mean(tf.abs(train_output-self.labels))
                else:
                    equal=tf.equal(tf.argmax(train_output,1),tf.argmax(self.labels,1))
                    train_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
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
                    for j in range(batches):
                        index1=j*batch
                        index2=(j+1)*batch
                        if batch!=1:
                            data_batch=self.train_data[index1:index2]
                            labels_batch=self.train_labels[index1:index2]
                        else:
                            data_batch=self.train_data[j]
                            labels_batch=self.train_labels[j]
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
                        data_batch=np.concatenate((self.train_data[index1:],self.train_data[:index2]))
                        labels_batch=np.concatenate((self.train_labels[index1:],self.train_labels[:index2]))
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
                        self.test_accuracy_list.append(self.test_accuracy)
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
                        self.test_accuracy_list.append(self.test_accuracy)
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
            if model_path!=None:
                self.save(model_path)
            self.time=self.time-int(self.time)
            if self.time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            print()
            print('last loss:{0:.6f}'.format(self.train_loss))
            if len(self.labels_shape)==2:
                print('accuracy:{0:.1f}%'.format(self.train_accuracy*100))
            else:
                print('accuracy:{0:.6f}'.format(self.train_accuracy))
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
            print('time:{0}s'.format(self.time))
            return
    
    
    def end(self):
        with self.graph.as_default():
            self.end_flag=True
            self.continue_train=False
            self.last_weight=self.sess.run(self.weight)
            self.last_bias=self.sess.run(self.bias)
            self.weight.clear()
            self.bias.clear()
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
            return test_loss,test_accuracy
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.total_epoch))
        print()
        print('l2:{0}'.format(self.l2))
        print()
        print('dropout:{0}'.format(self.dropout))
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
        if len(self.labels_shape)==2:
            print('train accuracy:{0:.1f}%'.format(self.train_accuracy*100))
        else:
            print('train accuracy:{0:.6f}'.format(self.train_accuracy))
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        print()
        if len(self.labels_shape)==2:
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
        if len(self.labels_shape)==2:
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
        if len(self.labels_shape)==2:
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
            if len(self.labels_shape)==2:
                print('test accuracy:{0:.1f}%'.format(self.test_accuracy*100))
            else:
                print('test accuracy:{0:.6f}'.format(self.test_accuracy))
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
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump([self.last_weight,self.last_bias],parameter_file)
        parameter_file.close()
        return


    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.last_weight,output_file)
        pickle.dump(self.last_bias,output_file)
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.data_shape,output_file)
        pickle.dump(self.labels_shape,output_file)
        pickle.dump(self.data_dtype,output_file)
        pickle.dump(self.labels_dtype,output_file)
        pickle.dump(self.hidden_layers,output_file)
        pickle.dump(self.function,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.l2,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.dropout,output_file)
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
        self.last_weight=pickle.load(input_file)
        self.last_bias=pickle.load(input_file)
        self.shape0=pickle.load(input_file)
        self.data_shape=pickle.load(input_file)
        self.labels_shape=pickle.load(input_file)
        self.data_dtype=pickle.load(input_file)
        self.labels_dtype=pickle.load(input_file)
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.data=tf.placeholder(dtype=self.data_dtype,shape=[None,None],name='data')
            if len(self.labels_shape)==2:
                self.labels=tf.placeholder(dtype=self.labels_dtype,shape=[None,None],name='labels')
            else:
                self.labels=tf.placeholder(dtype=self.labels_dtype,shape=[None],name='labels')
        self.hidden_layers=pickle.load(input_file)
        self.function=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.l2=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.dropout=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_accuracy=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_accuracy_list=pickle.load(input_file)
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
                    return output
                else:
                    softmax=np.sum(np.exp(output),axis=1).reshape(output.shape[0],1)
                    softmax=np.exp(output)/softmax
                    output=np.argmax(softmax,axis=1)+1
                    if save_path!=None:
                        output_file=open(save_path,'wb')
                        pickle.dump(output,output_file)
                        output_file.close()
                    return output
    
    
    def predicate(self,data,save_path=None,processor=None):
        with self.graph.as_default():
            if processor!=None:
                self.processor=processor
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
            return output
