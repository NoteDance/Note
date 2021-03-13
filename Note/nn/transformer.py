import tensorflow as tf
import Note.create.nn as n
from tensorflow.python.ops import state_ops
import tensorflow.keras.optimizers as optimizers
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class transformer:
    def __init__(self,train_data=None,train_labels=None,test_data=None,test_labels=None):
        with tf.name_scope('data'):
            self.train_data=train_data
            self.train_labels=train_labels
            self.test_data=test_data
            self.test_labels=test_labels
            self.shape0=train_data.shape[0]
        with tf.name_scope('parameter'):
            self.embedding_w=[]
            self.qw1=[]
            self.kw1=[]
            self.vw1=[]
            self.fw1=[]
            self.qw2=[]
            self.kw2=[]
            self.vw2=[]
            self.qw3=[]
            self.kw3=[]
            self.vw3=[]
            self.fw2=[]
            self.ow=None
        with tf.name_scope('hyperparameter'):
            self.batch=None
            self.epoch=0
            self.lr=None
            self.layers=None
        self.optimizer=None
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.buffer_size=None
        self.total_epoch=0
        self.time=0
        self.total_time=0
        self.processor='GPU:0'
    
    
    def weight_init(self,shape,mean,stddev):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
            
    
    def bias_init(self,shape,mean,stddev):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
    
    
    def structure(self,embedding_w,word_size,layers=6,mean=0,stddev=0.07,dtype=np.float32):
        self.test_flag=False
        self.train_loss_list.clear()
        self.train_accuracy_list.clear()
        self.test_loss_list.clear()
        self.test_acc_list.clear()
        self.dtype=dtype
        with tf.name_scope('hyperparameter'):
            self.layers=layers
        self.epoch=0
        self.total_epoch=0
        self.time=0
        self.total_time=0
        with tf.name_scope('parameter_initialization'):
            self.embedding_w=embedding_w
            if self.ow==None:
                if self.embedding_w==None:
                    self.embedding_w=self.weight_init(shape=[self.train_data.shape[2],512],mean=mean,stddev=stddev)
                for i in range(self.h):
                    self.qw1.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.kw1.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.vw1.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.fw1.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.qw2.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.kw2.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.vw2.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.qw3.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.kw3.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.vw3.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    self.fw2.append(self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                self.ow=self.weight_init(shape=[512,word_size],mean=mean,stddev=stddev)
            else:
                if self.embedding_w==None:
                    state_ops.assign(self.embedding_w,self.weight_init(shape=[self.train_data.shape[2],512],mean=mean,stddev=stddev))
                for i in range(self.h):
                    state_ops.assign(self.qw1[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.kw1[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.vw1[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.fw1[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.qw2[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.kw2[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.vw2[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.qw3[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.kw3[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.vw3[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                    state_ops.assign(self.fw2[i],self.weight_init(shape=[512,512],mean=mean,stddev=stddev))
                state_ops.assign(self.ow,self.weight_init(shape=[512,word_size],mean=mean,stddev=stddev))
        return
    
    
    def encoder(self,word_vector):
        encoder_temp=[]
        encoder=[x for x in range(self.layers)]
        if len(word_vector.shape)==2:
            word_vector=tf.reshape(word_vector,shape=[1,word_vector.shape[0],word_vector.sahpe[1]])
        for i in range(self.layers):
            with tf.device(self._processor[i]):
                with tf.name_scope('self_attention'):
                    query=tf.einsum('ijk,kl->ijl',word_vector,self.qw1[i])
                    key=tf.einsum('ijk,kl->ijl',word_vector,self.kw1[i])
                    value=tf.einsum('ijk,kl->ijl',word_vector,self.vw1[i])
                    query=tf.reshape(query,shape=[query.shape[0],query.shape[1],8,64])
                    key=tf.reshape(key,shape=[64,8,key.shape[1],key.shape[0]])
                    value=tf.reshape(value,shape=[value.shape[0],value.shape[1],8,64])
                    _value=tf.zeros(shape=[value.shape[0],value.shape[1]])
                    for j in range(8):
                        for k in range(query.shape[0]):
                            qk=tf.matmul(query[k,:,j,:],key[:,j,:,k])
                            qk=qk/8
                            softmax=tf.nn.softmax(qk,axis=1)
                            softmax=tf.reshape(softmax,shape=[-1,1])
                            for l in range(query.shape[1]):
                                _value[k][l]=tf.reduce_sum(softmax[l]*value[k,:,j,:],aixs=0)
                        encoder_temp.append(_value)
                    encoder[i]=tf.concat(encoder_temp,axis=2)
                    encoder_temp.clear()
                    with tf.name_scope('fnn1'):
                        word_vector=tf.nn.relu(tf.einsum('ijk,kl->ijl',encoder[i],self.fw1[i]))+word_vector
                        with tf.name_scope('normalization'):
                            mean=tf.reduce_mean(word_vector,axis=2)
                            var=(word_vector-tf.reshape(mean,shape=[mean.shape[0],mean.shape[1],1]))**2
                            word_vector=(word_vector-mean)/tf.sqrt(var+1e-07)
                    encoder[i]=word_vector
        return encoder
    
    
    def decoder(self,word_vector,encoder):
        decoder_temp=[]
        decoder1=[x for x in range(self.layers)]
        decoder2=[x for x in range(self.layers)]
        for i in range(self.layers):
            with tf.device(self._processor[i]):
                with tf.name_scope('self_attention1'):
                    query1=tf.einsum('ijk,kl->ijl',word_vector,self.qw2[i])
                    mask=tf.constant(np.triu(np.zeros([query1.shape[1],query1.shape[1]])-1e10))
                    key1=tf.einsum('ijk,kl->ijl',word_vector,self.kw2[i])
                    value1=tf.einsum('ijk,kl->ijl',word_vector,self.vw2[i])
                    query1=tf.reshape(query1,shape=[query1.shape[0],query1.shape[1],8,64])
                    key1=tf.reshape(key1,shape=[64,8,key1.shape[1],key1.shape[0]])
                    value1=tf.reshape(value1,shape=[value1.shape[0],value1.shape[1],8,64])
                    _value=tf.zeros(shape=[value1.shape[0],value1.shape[1]])
                    for j in range(8):
                        for k in range(query1.shape[0]):
                            qk1=tf.matmul(query1[k,:,j,:],key1[:,j,:,k])
                            qk1=qk1/8
                            qk1=qk1*mask
                            softmax=tf.nn.softmax(qk1,axis=1)
                            softmax=tf.reshape(softmax,shape=[-1,1])
                            for l in range(query1.shape[1]):
                                _value[k][l]=tf.reduce_sum(softmax[l]*value1[k,:,j,:],aixs=0)
                        decoder_temp.append(_value)
                        decoder_temp.clear()
                    decoder1[i]=tf.concat(decoder_temp,axis=2)
                with tf.name_scope('self_attention2'):
                    query2=tf.einsum('ijk,kl->ijl',decoder1[i],self.qw3[i])
                    key2=tf.einsum('ijk,kl->ijl',encoder[i],self.kw3[i])
                    value2=tf.einsum('ijk,kl->ijl',encoder[i],self.vw3[i])
                    query2=tf.reshape(query2,shape=[query2.shape[0],query2.shape[1],8,64])
                    key2=tf.reshape(key2,shape=[64,8,key2.shape[1],key2.shape[0]])
                    value2=tf.reshape(value2,shape=[value2.shape[0],value2.shape[1],8,64])
                    for j in range(8):
                        for k in range(query2.shape[0]):
                            qk2=tf.matmul(query2[k,:,j,:],key2[:,j,:,k])
                            qk2=qk2/8
                            qk2=qk2
                            softmax=tf.nn.softmax(qk2,axis=1)
                            softmax=tf.reshape(softmax,shape=[-1,1])
                            for l in range(query2.shape[1]):
                                _value[k][l]=tf.reduce_sum(softmax[l]*value2[k,:,j,:],aixs=0)
                        decoder_temp.append(_value)
                        decoder2[i]=tf.concat(decoder_temp,axis=2)
                        with tf.name_scope('fnn2'):
                            word_vector=tf.nn.relu(tf.einsum('ijk,kl->ijl',decoder2[i],self.fw2[i]))+word_vector
                            with tf.name_scope('normalization'):
                                mean=tf.reduce_mean(word_vector,axis=2)
                                var=(word_vector-tf.reshape(mean,shape=[mean.shape[0],mean.shape[1],1]))**2
                                word_vector=(word_vector-mean)/tf.sqrt(var+1e-07)
        return word_vector
                                        
            
    @tf.function       
    def forward_propagation(self,train_data,train_labels):
        with tf.name_scope('processor_allocation'):
            self._processor=[x for x in range(self.layers)]
            if type(self.processor)==list:
                for i in range(self.layers):
                    self._processor[i]=self.processor[i]
            else:
                for i in range(self.layers):
                    self._processor[i]=self.processor     
        with tf.name_scope('forward_propagation'):
            with tf.device(self._processor[0]):
                with tf.name_scope('embedding'):
                    if len(self.embedding_w)==2:
                        word_vector1=tf.einsum('ijk,kl->ijl',train_data,self.embedding_w[0])+tf.einsum('ijk,kl->ijl',train_data,self.embedding_w[1])
                        word_vector2=tf.einsum('ijk,kl->ijl',train_labels,self.embedding_w[0])+tf.einsum('ijk,kl->ijl',train_labels,self.embedding_w[1])
                    elif type(self.embedding_w)!=list:
                        word_vector1=tf.einsum('ijk,kl->ijl',train_data,self.embedding_w)
                        word_vector2=tf.einsum('ijk,kl->ijl',train_labels,self.embedding_w)
                    else:
                        word_vector1=tf.einsum('ijk,kl->ijl',train_data,self.embedding_w[0])
                        word_vector2=tf.einsum('ijk,kl->ijl',train_labels,self.embedding_w[0])
                arange1=tf.constant(np.arange(word_vector1.shape[1]))
                arange2=tf.constant(np.arange(word_vector1.shape[2]))
                arange3=tf.constant(np.arange(word_vector2.shape[1]))
                arange4=tf.constant(np.arange(word_vector2.shape[2]))
                word_vector1=word_vector1+tf.math.sin(arange1/10000**(arange2/512)*((arange2*2)%2==0)+tf.math.cos(arange1/10000**(arange2/512))*((arange2*2+1)%2!=0))
                word_vector2=word_vector2+tf.math.sin(arange3/10000**(arange4/512)*((arange4*2)%2==0)+tf.math.cos(arange3/10000**(arange4/512))*((arange4*2+1)%2!=0))
            with tf.name_scope('encoder'):
                encoder=self.encoder(self,word_vector1)
            with tf.name_scope('decoder'):
                output=self.decoder(self,word_vector2,encoder)
            return output
                    
                                
    def train(self,batch=None,epoch=None,lr=0.001,test=False,test_batch=None,model_path=None,one=True,processor=None,buffer_size=None):
        with tf.name_scope('hyperparameter'):
            self.batch=batch
            self.epoch=0
            self.lr=lr
        batches=int((self.shape0-self.shape0%batch)/batch)
        if self.shape0%batch!=0:
            batches+=1
        if buffer_size!=None:
            self.buffer_size=buffer_size
        elif self.buffer_size!=None:
            pass
        else:
            self.buffer_size=self.shape0
        self.time=0
        self.test_flag=test
        if processor!=None:
            self.processor=processor
        with tf.name_scope('variable'):
            variable=[self.qw1,self.kw1,self.vw1,self.fw1,self.qw2,self.kw2,self.vw2,self.qw3,self.kw3,self.vw3,self.fw2]
            variable=n.extend(variable)
        with tf.name_scope('optimizer'):
            self.optimizer='Adam'
            optimizer=optimizers.Adam(lr)
        if self.total_epoch==0:
            epoch=epoch+1
        for i in range(epoch):
            t1=time.time()
            if batch!=None:
                total_loss=0
                total_acc=0
                train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).shuffle(self.buffer_size).batch(batch)
                for data_batch,labels_batch in train_ds:
                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            output=self.forward_propagation(data_batch)
                            batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels_batch))
                    if i==0 and self.total_epoch==0:
                        batch_loss=batch_loss.numpy()
                    else:
                        with tf.name_scope('apply_gradient'):
                            n.apply_gradient(tape,optimizer,batch_loss,variable)
                    total_loss+=batch_loss
                    with tf.name_scope('accuracy'):
                        batch_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(labels_batch,2)!=0,tf.int32)==tf.argmax(labels_batch,2),tf.float32))
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                loss=total_loss/batches
                train_acc=total_acc/batches
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                self.train_acc_list.append(train_acc.astype(np.float32))
                self.train_acc=train_acc
                self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_acc_list.append(self.test_acc)
            else:
                with tf.GradientTape() as tape:
                    with tf.name_scope('forward_propagation/loss'):
                        output=self.forward_propagation(self.train_data)
                        train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=self.train_labels))
                if i==0 and self.total_epoch==0:
                    loss=train_loss.numpy()
                else:
                    with tf.name_scope('apply_gradient'):
                        n.apply_gradient(tape,optimizer,batch_loss,variable)
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                with tf.name_scope('accuracy'):
                    train_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(self.train_labels,2)!=0,tf.int32)==tf.argmax(self.train_labels,2),tf.float32))
                acc=train_acc.numpy()
                self.train_acc_list.append(acc.astype(np.float32))
                self.train_acc=acc
                self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_acc_list.append(self.test_acc)
            self.epoch+=1
            self.total_epoch+=1
            if epoch%10!=0:
                d=epoch-epoch%10
                d=int(d/10)
            else:
                d=epoch/10
            if d==0:
                d=1
            if i%d==0:
                if self.total_epoch==0:
                    print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                else:
                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                if model_path!=None and i%epoch*2==0:
                    self.save(model_path,i,one)
            t2=time.time()
            self.time+=(t2-t1)
        if model_path!=None:
            self.save(model_path)
        self.qw1=[variable[:self.h]]
        self.time=self.time-int(self.time)
        if self.time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(self.train_loss))
        print('accuracy:{0:.1f}%'.format(self.train_acc*100))
        print('time:{0}s'.format(self.time))
        return
        
    
    def test(self,test_data,test_labels,batch=None,buffer_size=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
            if test_data.shape[0]%batch!=0:
                batches+=1
            if buffer_size!=None:
                buffer_size=buffer_size
            else:
                buffer_size=len(test_data)
            test_ds=tf.data.Dataset.from_tensor_slices((test_data,test_labels)).batch(batch)
            for data_batch,labels_batch in test_ds:
                with tf.name_scope('loss'):
                     output=self.forward_propagation(data_batch)
                     batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels_batch))
                total_loss+=batch_loss.numpy()
                with tf.name_scope('accuracy'):
                    batch_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(labels_batch,2)!=0,tf.int32)==tf.argmax(labels_batch,2),tf.float32))
                total_acc+=batch_acc.numpy()
            test_loss=total_loss/batches
            test_acc=total_acc/batches
            test_loss=test_loss
            test_acc=test_acc
            test_loss=test_loss.astype(np.float32)
            test_acc=test_acc.astype(np.float32)
        else:
            with tf.name_scope('loss'):
                output=self.forward_propagation(test_data)
                test_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=test_labels))
            with tf.name_scope('accuracy'):
                test_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(test_labels,2)!=0,tf.int32)==tf.argmax(test_labels,2),tf.float32))
            test_loss=test_loss.numpy().astype(np.float32)
            test_acc=test_acc.numpy().astype(np.float32)
        return test_loss,test_acc
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.total_epoch))
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
        print('train acc:{0:.1f}%'.format(self.train_acc*100))
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        print('test acc:{0:.1f}%'.format(self.test_acc*100))
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
        plt.plot(np.arange(self.total_epoch),self.train_acc_list)
        plt.title('train acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('train loss:{0:.6f}'.format(self.train_loss))
        print('train acc:{0:.1f}%'.format(self.train_acc*100))
        return
    
    
    def test_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.test_acc_list)
        plt.title('test acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('test loss:{0:.6f}'.format(self.test_loss))
        print('test acc:{0:.1f}%'.format(self.test_acc*100))   
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
        plt.plot(np.arange(self.total_epoch),self.train_acc_list,'b-',label='train acc')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch),self.test_acc_list,'r-',label='test acc')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        print('train loss:{0}'.format(self.train_loss))
        print('train acc:{0:.1f}%'.format(self.train_acc*100))       
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            print('test acc:{0:.1f}%'.format(self.test_acc*100))
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump([self.embedding_w,self.qw1,self.kw1,self.vw1,self.fw1,self.qw2,self.kw2,self.vw2,self.qw3,self.kw3,self.vw3,self.fw2,self.ow],parameter_file)
        parameter_file.close()
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'\save.dat','wb')
            path=path+'\save.dat'
            index=path.rfind('\\')
            parameter_file=open(path.replace(path[index+1:],'parameter.dat'),'wb')
        else:
            output_file=open(path+'\save-{0}.dat'.format(i+1),'wb')
            path=path+'\save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            parameter_file=open(path.replace(path[index+1:],'parameter-{0}.dat'.format(i+1)),'wb')
        with tf.name_scope('save_parameter'):  
            pickle.dump([self.embedding_w,self.qw1,self.kw1,self.vw1,self.fw1,self.qw2,self.kw2,self.vw2,self.qw3,self.kw3,self.vw3,self.fw2,self.ow],parameter_file)
        with tf.name_scope('save_hyperparameter'):
            pickle.dump(self.batch,output_file)
            pickle.dump(self.lr,output_file)
            pickle.dump(self.layers,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_acc,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_acc_list,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_acc,output_file)
            pickle.dump(self.test_loss_list,output_file)
            pickle.dump(self.test_acc_list,output_file)
        pickle.dump(self.buffer_size,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        parameter_file.close()
        return
    

    def restore(self,s_path,p_path):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        parameter=pickle.load(parameter_file)
        with tf.name_scope('restore_parameter'):
            self.embedding_w=parameter[0]
            self.qw1=parameter[1]
            self.kw1=parameter[2]
            self.vw1=parameter[3]
            self.fw1=parameter[4]
            self.qw2=parameter[5]
            self.kw2=parameter[6]
            self.vw2=parameter[7]
            self.qw3=parameter[8]
            self.kw3=parameter[9]
            self.vw3=parameter[10]
            self.fw2=parameter[11]
            self.ow=parameter[12]
        with tf.name_scope('restore_hyperparameter'):
            self.batch=pickle.load(input_file)
            self.lr=pickle.load(input_file)
            self.layers=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.shape0=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_acc=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_acc_list=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_acc=pickle.load(input_file)
            self.test_loss_list=pickle.load(input_file)
            self.test_acc_list=pickle.load(input_file)
        self.buffer_size=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        input_file.close()
        parameter_file.close()
        return
