import tensorflow as tf
import Note.create.TF2 as TF2
from tensorflow.python.ops import state_ops
import tensorflow.keras.optimizers as optimizers
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class transformer:
    def __init__(self,train_data=None,train_labels=None,test_data=None,test_labels=None):
        self.tf2=TF2.tf2()
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
        self.hyperparameter=None
        self.regulation=None
        self.optimizer=None
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.total_epoch=0
        self.time=0
        self.total_time=0
        self.processor='GPU:0'
        
    
    def weight_init(self,shape,mean,stddev):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
            
    
    def bias_init(self,shape,mean,stddev):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
    
    
    def structure(self,embedding_w,word_size,layers=6,mean=0,stddev=0.07,dtype=np.float32):
        self.epoch=0
        self.total_epoch=0
        self.test_flag=False
        self.train_loss_list.clear()
        self.train_accuracy_list.clear()
        self.test_loss_list.clear()
        self.test_acc_list.clear()
        self.dtype=dtype
        with tf.name_scope('hyperparameter'):
            self.layers=layers
        self.hyperparameter={'layers':'encoder-decoder layers'}
        self.time=None
        self.total_time=None
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
        
        
    def batch(self,data):
        if self.index1==self.batches*self.batch:
            return np.concatenate([data[self.index1:],data[:self.index2]])
        else:
            return data[self.index1:self.index2]
        
        
    def extend(self,variable):
        for i in range(len(variable)-1):
            variable[0].extend[variable[i+1]]
        return variable[0]
    
    
    def apply_gradient(self,tape,optimizer,loss,variable):
        gradient=tape.gradient(loss,variable)
        optimizer.apply_gradients(zip(gradient,variable))
        return
                        
                                
    def train(self,batch=None,epoch=None,lr=0.001,test=False,test_batch=None,model_path=None,one=True,processor=None):
        with tf.name_scope('hyperparameter'):
            self.batch=batch
            self.lr=lr
        self.test_flag=test
        if processor!=None:
            self.processor=processor
        with tf.name_scope('variable'):
            variable=[self.qw1,self.kw1,self.vw1,self.fw1,self.qw2,self.kw2,self.vw2,self.qw3,self.kw3,self.vw3,self.fw2]
            variable=self.extend(variable)
        with tf.name_scope('optimizer'):
            self.optimizer=['Adam',{'lr':lr}]
            optimizer=optimizers.Adam(lr)
        if self.total_epoch==0:
            epoch=epoch+1
        t1=time.time()
        for i in range(epoch):
            if batch!=None:
                batches=int((self.shape0-self.shape0%batch)/batch)
                self.tf2.batches=batches
                total_loss=0
                total_acc=0
                random=np.arange(self.shape0)
                np.random.shuffle(random)
                with tf.name_scope('randomize_data'):
                    train_data=self.train_data[random]
                    train_labels=self.train_labels[random]
                for j in range(batches):
                    self.tf2.index1=j*batch
                    self.tf2.index2=(j+1)*batch
                    with tf.name_scope('data_batch'):
                        train_data_batch=self.tf2.batch(train_data)
                        train_labels_batch=self.tf2.batch(train_labels)
                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            output=self.forward_propagation(train_data_batch)
                            batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=train_labels_batch))
                        if i==0 and self.total_epoch==0:
                            batch_loss=batch_loss.numpy()
                        else:
                            with tf.name_scope('apply_gradient'):
                                self.tf2.apply_gradient(tape,optimizer,batch_loss,variable)
                    total_loss+=batch_loss
                    with tf.name_scope('accuracy'):
                        batch_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(train_labels_batch,2)!=0,tf.int32)==tf.argmax(train_labels_batch,2),tf.float32))
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                if self.shape0%batch!=0:
                    batches+=1
                    self.tf2.batches+=1
                    self.tf2.index1=batches*batch
                    self.tf2.index2=batch-(self.shape0-batches*batch)
                    with tf.name_scope('data_batch'):
                        train_data_batch=self.tf2.batch(train_data)
                        train_labels_batch=self.tf2.batch(train_labels)
                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            output=self.forward_propagation(train_data_batch)
                            batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=train_labels_batch))
                        if i==0 and self.total_epoch==0:
                            batch_loss=batch_loss.numpy()
                        else:
                            with tf.name_scope('apply_gradient'):
                                self.tf2.apply_gradient(tape,optimizer,batch_loss,variable)
                    total_loss+=batch_loss
                    with tf.name_scope('accuracy'):
                        batch_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(train_labels_batch,2)!=0,tf.int32)==tf.argmax(train_labels_batch,2),tf.float32))
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                loss=total_loss/batches
                train_acc=total_acc/batches
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                self.train_acc_list.append(float(train_acc))
                self.train_acc=train_acc
                self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(self.tst_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_acc_list.append(self.test_acc)
            else:
                random=np.arange(self.shape0)
                np.random.shuffle(random)
                with tf.name_scope('randomize_data'):
                    train_data=self.train_data[random]
                    train_labels=self.train_labels[random]
                with tf.GradientTape() as tape:
                    with tf.name_scope('forward_propagation/loss'):
                        output=self.forward_propagation(train_data)
                        train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=train_labels))
                    if i==0 and self.total_epoch==0:
                        loss=train_loss.numpy()
                    else:
                        with tf.name_scope('apply_gradient'):
                            self.tf2.apply_gradient(tape,optimizer,batch_loss,variable)
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                with tf.name_scope('accuracy'):
                    train_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(train_labels,2)!=0,tf.int32)==tf.argmax(train_labels,2),tf.float32))
                acc=train_acc.numpy()
                self.train_acc_list.append(float(acc))
                self.train_acc=acc
                self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(self.tst_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_acc_list.append(self.test_acc)
            if epoch%10!=0:
                temp=epoch-epoch%10
                temp=int(temp/10)
            else:
                temp=epoch/10
            if temp==0:
                temp=1
            if i%temp==0:
                if self.total_epoch==0:
                    print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                else:
                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                if model_path!=None and i%epoch*2==0:
                    self.save(model_path,i,one)
        t2=time.time()
        _time=(t2-t1)-int(t2-t1)
        if _time<0.5:
            self.time=int(t2-t1)
        else:
            self.time=int(t2-t1)+1
        self.total+=self.time
        print()
        print('last loss:{0:.6f}'.format(self.train_loss))
        print('accuracy:{0:.1f}%'.format(self.train_acc*100))
        if self.total_epoch==0:
            self.total_epoch=epoch-1
            self.epoch=epoch-1
        else:
            self.total_epoch=self.total_epoch+epoch
            self.epoch=epoch
        print('time:{0}s'.format(self.time))
        return
        
    
    def test(self,test_data,test_labels,batch=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
            self.tf2.batches=batches
            for j in range(batches):
                self.tf2.index1=j*batch
                self.tf2.index2=(j+1)*batch
                with tf.name_scope('data_batch'):
                    test_data_batch=self.tf2.batch(test_data)
                    test_labels_batch=self.tf2.batch(test_labels)
                with tf.name_scope('loss'):
                     output=self.forward_propagation(test_data_batch)
                     batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=test_labels_batch))
                total_loss+=batch_loss.numpy()
                with tf.name_scope('accuracy'):
                    batch_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(test_labels_batch,2)!=0,tf.int32)==tf.argmax(test_labels_batch,2),tf.float32))
                total_acc+=batch_acc.numpy()
            if test_data.shape[0]%batch!=0:
                batches+=1
                self.tf2.batches+=1
                self.tf2.index1=batches*batch
                self.tf2.index2=batch-(self.shape0-batches*batch)
                with tf.name_scope('data_batch'):
                    test_data_batch=self.batch(test_data)
                    test_labels_batch=self.batch(test_labels)
                with tf.name_scope('loss'):
                    output=self.forward_propagation(test_data_batch)
                    batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=test_labels_batch))
                total_loss+=batch_loss.numpy()
                with tf.name_scope('accuracy'):
                    batch_acc=tf.reduce_mean(tf.cast(tf.argmax(output,2)*tf.cast(tf.argmax(test_labels_batch,2)!=0,tf.int32)==tf.argmax(test_labels_batch,2),tf.float32))
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
        print('epoch:{0}'.format(self.epoch))
        print()
        print('optimizer:{0}'.format(self.optimizer))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.time))
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
        plt.plot(np.arange(self.epoch+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.train_acc_list)
        plt.title('train acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('train loss:{0:.6f}'.format(self.train_loss))
        print('train acc:{0:.1f}%'.format(self.train_acc*100))
        return
    
    
    def test_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.test_acc_list)
        plt.title('test acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('test loss:{0:.6f}'.format(self.test_loss))
        print('test acc:{0:.1f}%'.format(self.test_acc*100))   
        return 
    
        
    def comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(self.epoch+1),self.test_loss_list,'r-',label='test loss')
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.train_acc_list,'b-',label='train acc')
        if self.test_flag==True:
            plt.plot(np.arange(self.epoch+1),self.test_acc_list,'r-',label='test acc')
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
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        with tf.name_scope('save_parameter'):  
            pickle.dump(self.embedding_w,output_file)
            pickle.dump(self.qw1,output_file)
            pickle.dump(self.kw1,output_file)
            pickle.dump(self.vw1,output_file)
            pickle.dump(self.fw1,output_file)
            pickle.dump(self.qw2,output_file)
            pickle.dump(self.kw2,output_file)
            pickle.dump(self.vw2,output_file)
            pickle.dump(self.qw3,output_file)
            pickle.dump(self.kw3,output_file)
            pickle.dump(self.vw3,output_file)
            pickle.dump(self.fw2,output_file)
            pickle.dump(self.ow,output_file)
        with tf.name_scope('save_hyperparameter'):
            pickle.dump(self.batch,output_file)
            pickle.dump(self.epoch,output_file)
            pickle.dump(self.lr,output_file)
            pickle.dump(self.layers,output_file)
        pickle.dump(self.hyperparameter,output_file)
        pickle.dump(self.regulation,output_file)
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
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.time,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        input_file=open(model_path,'rb')
        with tf.name_scope('restore_parameter'):
            self.embedding_w=pickle.load(input_file)
            self.qw1=pickle.load(input_file)
            self.kw1=pickle.load(input_file)
            self.vw1=pickle.load(input_file)
            self.fw1=pickle.load(input_file)
            self.qw2=pickle.load(input_file)
            self.kw2=pickle.load(input_file)
            self.vw2=pickle.load(input_file)
            self.qw3=pickle.load(input_file)
            self.kw3=pickle.load(input_file)
            self.vw3=pickle.load(input_file)
            self.fw2=pickle.load(input_file)
            self.ow=pickle.load(input_file)
        with tf.name_scope('restore_hyperparameter'):
            self.batch=pickle.load(input_file)
            self.epoch=pickle.load(input_file)
            self.lr=pickle.load(input_file)
            self.layers=pickle.load(input_file)
        self.hyperparameter=pickle.load(input_file)
        self.regulation=pickle.load(input_file)
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
        self.epoch=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        input_file.close()
        return