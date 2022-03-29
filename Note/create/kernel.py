import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class kernel:
    def __init__(self,nn=None):
        if nn!=None:
            self.nn=nn
        self.PO=None
        self.thread_lock=None
        self.thread=None
        self.ol=None
        self.batch=None
        self.epoch=0
        self.opt=None
        self.sopt=None
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.eflag=None
        self.bflag=None
        self.optf=None
        self.acc_flag1=None
        self.acc_flag2=None
        self.flag=None
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.test=False
        self.total_epoch=0
        self.time=0
        self.total_time=0
    
    
    def data(self,train_data,train_labels,test_data=None,test_labels=None):
        self.train_data=train_data
        self.train_labels=train_labels
        if type(train_data)==list:
            self.data_batch=[x for x in range(len(train_data))]
        if type(train_labels)==list:
            self.labels_batch=[x for x in range(len(train_labels))]
        self.test_data=test_data
        self.test_labels=test_labels
        if test_data!=None:
            self.test=True
        if type(self.train_data)==list:
            self.shape0=train_data[0].shape[0]
        else:
            self.shape0=train_data.shape[0]
        if self.thread!=None:
            self.t=-np.arange(-self.thread,1)
            if self.PO!=True:
                self.train_loss=np.zeros(self.thread)
                self.train_acc=np.zeros(self.thread)
                self.train_loss_list=[[] for _ in range(self.thread)]
                self.train_acc_list=[[] for _ in range(self.thread)]
            if test_data!=None:
                if self.PO!=True:
                    self.test_loss=np.zeros(self.thread)
                    self.test_acc=np.zeros(self.thread)
                    self.test_loss_list=[[] for _ in range(self.thread)]
                    self.test_acc_list=[[] for _ in range(self.thread)]
            self.epoch=np.zeros(self.thread)
            self.total_epoch=np.zeros(self.thread)
            self.time=np.zeros(self.thread)
            self.total_time=np.zeros(self.thread)
        return
    
    
    def init(self,param=None):
        if param!=None:
            self.nn.param=param
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        self.test_loss_list.clear()
        self.test_acc_list.clear()
        self.test=False
        self.epoch=0
        self.total_epoch=0
        self.time=0
        self.total_time=0
        return
    
    
    def extend(self,thread):
        t=-np.arange(-thread,1)+self.thread+1
        self.t=t.extend(self.t)
        self.thread+=thread
        if self.PO!=True:
            self.train_loss=np.concatenate((self.train_loss,np.zeros(self.t)))
            self.train_acc=np.concatenate((self.train_acc,np.zeros(self.t)))
            self.train_loss_list.extend([[] for _ in range(len(self.t))])
            self.train_acc_list.extend([[] for _ in range(len(self.t))])
        if self.test==True:
            if self.PO!=True:
                self.test_loss=np.concatenate((self.test_loss,np.zeros(self.t)))
                self.test_acc=np.concatenate((self.test_acc,np.zeros(self.t)))
                self.test_loss_list.extend([[] for _ in range(len(self.t))])
                self.test_acc_list.extend([[] for _ in range(len(self.t))])
        self.epoch=np.concatenate((self.epoch,np.zeros(self.t)))
        self.total_epoch=np.concatenate((self.total_epoch,np.zeros(self.t)))
        self.time=np.concatenate((self.time,np.zeros(self.t)))
        self.total_time=np.concatenate((self.total_time,np.zeros(self.t)))
        return
    
    
    def set_end(self,end_loss=None,end_acc=None,end_test_loss=None,end_test_acc=None):
        if end_loss!=None:
            self.end_loss=end_loss
        if end_acc!=None:
            self.end_acc=end_acc
        if end_test_loss!=None:
            self.end_test_loss=end_test_loss
        if end_test_acc!=None:
            self.end_test_acc=end_test_acc
        return
    
    
    def apply_gradient(self,tape,opt,loss,parameter):
        gradient=tape.gradient(loss,parameter)
        opt.apply_gradients(zip(gradient,parameter))
        return
    
    
    def end(self):
        if self.end_loss!=None and self.train_loss<=self.end_loss:
            return True
        elif self.end_acc!=None and self.train_acc>=self.end_acc:
            return True
        elif self.end_loss!=None and self.end_acc!=None and self.train_loss<=self.end_loss and self.train_acc>=self.end_acc:
            return True
        elif self.end_test_loss!=None and self.test_loss<=self.end_test_loss:
            return True
        elif self.end_test_acc!=None and self.test_acc>=self.end_test_acc:
            return True
        elif self.end_test_loss!=None and self.end_test_acc!=None and self.test_loss<=self.end_test_loss and self.test_acc>=self.end_test_acc:
            return True
    
    
    def _train(self,batch=None,test_batch=None,data_batch=None,labels_batch=None,t=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            batches=int((self.shape0-self.shape0%batch)/batch)
            for j in range(batches):
                index1=j*batch
                index2=(j+1)*batch
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        if batch!=1:
                            data_batch[i]=self.train_data[i][index1:index2]
                        else:
                            data_batch[i]=self.train_data[i][j]
                else:
                    if batch!=1:
                        data_batch=self.train_data[index1:index2]
                    else:
                        data_batch=self.train_data[j]
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        if batch!=1:
                            labels_batch[i]=self.train_labels[i][index1:index2]
                        else:
                            labels_batch[i]=self.train_labels[i][j]
                else:
                    if batch!=1:
                        labels_batch=self.train_labels[index1:index2]
                    else:
                        labels_batch=self.train_labels[j]
                with tf.GradientTape() as tape:
                    if self.thread==None:
                        output=self.nn.fp(data_batch)
                    else:
                        output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                if self.optf!=True:
                    if self.thread==None:
                        self.apply_gradient(tape,self.opt,batch_loss,self.nn.param)
                    else:
                        self.apply_gradient(tape,self.opt[t],batch_loss,self.nn.param[t])
                else:
                    if self.thread==None:
                        gradient=tape.gradient(batch_loss,self.nn.param)
                        self.sopt(gradient,self.nn.param)
                    else:
                        gradient=tape.gradient(batch_loss,self.nn.param[t])
                        self.sopt(gradient,self.nn.param,t)
                if self.total_epoch==1:
                    batch_loss=batch_loss.numpy()
                total_loss+=batch_loss
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                if self.bflag==True:
                    if self.thread==None:
                        self.nn.batchcount=j
                    else:
                        self.nn.batchcount[t]=j
            if self.shape0%batch!=0:
                batches+=1
                index1=batches*batch
                index2=batch-(self.shape0-batches*batch)
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        data_batch[i]=tf.concat([self.train_data[i][index1:],self.train_data[i][:index2]])
                else:
                    data_batch=tf.concat([self.train_data[index1:],self.train_data[:index2]])
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=tf.concat([self.train_labels[i][index1:],self.train_labels[i][:index2]])
                else:
                    labels_batch=tf.concat([self.train_labels[index1:],self.train_labels[:index2]])
                with tf.GradientTape() as tape:
                    if self.thread==None:
                        output=self.nn.fp(data_batch)
                    else:
                        output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                if self.optf!=True:
                    if self.thread==None:
                        self.apply_gradient(tape,self.opt,batch_loss,self.nn.param)
                    else:
                        self.apply_gradient(tape,self.opt[t],batch_loss,self.nn.param[t])
                else:
                    if self.thread==None:
                        gradient=tape.gradient(batch_loss,self.nn.param)
                        self.sopt(gradient,self.param)
                    else:
                        gradient=tape.gradient(batch_loss,self.nn.param[t])
                        self.sopt(gradient,self.nn.param,t)
                if self.total_epoch==1:
                    batch_loss=batch_loss.numpy()
                total_loss+=batch_loss
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                if self.bflag==True:
                    if self.thread==None:
                        self.nn.batchcount+=1
                    else:
                        self.nn.batchcount[t]+=1
            loss=total_loss/batches
            if self.acc_flag1==1:
                train_acc=total_acc/batches
            if self.thread==None:
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
            else:
                self.train_loss_list[t].append(loss.astype(np.float32))
                self.train_loss[t]=loss
                self.train_loss[t]=self.train_loss[t].astype(np.float32)
            if self.acc_flag1==1:
                if self.thread==None:
                    self.train_acc_list.append(train_acc.astype(np.float32))
                    self.train_acc=train_acc
                    self.train_acc=self.train_acc.astype(np.float32)
                else:
                    self.train_acc_list[t].append(train_acc.astype(np.float32))
                    self.train_acc[t]=train_acc
                    self.train_acc[t]=self.train_acc[t].astype(np.float32)
            if self.test==True:
                if self.thread==None:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    if self.acc_flag1==1:
                        self.test_acc_list.append(self.test_acc)
                else:
                    self.test_loss[t],self.test_acc[t]=self.test(self.test_data,self.test_labels,test_batch,t)
                    self.test_loss_list[t].append(self.test_loss[t])
                    if self.acc_flag1==1:
                        self.test_acc_list[t].append(self.test_acc[t])
        elif self.ol==None:
            with tf.GradientTape() as tape:
                if self.thread==None:
                    output=self.nn.fp(self.train_data)
                else:
                    output=self.nn.fp(data_batch,t)
                train_loss=self.nn.loss(output,self.train_labels)
            if self.optf!=True:
                if self.thread==None:
                    self.apply_gradient(tape,self.opt,train_loss,self.nn.param)
                else:
                    self.apply_gradient(tape,self.opt[t],batch_loss,self.nn.param[t])
            else:
                if self.thread==None:
                    gradient=tape.gradient(train_loss,self.nn.param)
                    self.sopt(gradient,self.nn.param)
                else:
                    gradient=tape.gradient(batch_loss,self.nn.param[t])
                    self.sopt(gradient,self.nn.param,t)
            if self.total_epoch==1:
                loss=train_loss.numpy()
            if self.thread==None:
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
            else:
                self.train_loss_list[t].append(loss.astype(np.float32))
                self.train_loss[t]=loss
                self.train_loss[t]=self.train_loss[t].astype(np.float32)
            if self.acc_flag1==1:
                if self.thread==None:
                    acc=self.nn.accuracy(output,self.train_labels)
                    acc=acc.numpy()
                    self.train_acc_list.append(acc.astype(np.float32))
                    self.train_acc=acc
                    self.train_acc=self.train_acc.astype(np.float32)
                else:
                    acc=self.nn.accuracy(output,self.train_labels[t])
                    acc=acc.numpy()
                    self.train_acc_list[t].append(acc.astype(np.float32))
                    self.train_acc[t]=acc
                    self.train_acc[t]=self.train_acc[t].astype(np.float32)
            if self.test==True:
                if self.thread==None:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    if self.acc_flag1==1:
                        self.test_acc_list.append(self.test_acc)
                else:
                    self.test_loss[t],self.test_acc[t]=self.test(self.test_data,self.test_labels,test_batch,t)
                    self.test_loss_list[t].append(self.test_loss[t])
                    if self.acc_flag1==1:
                        self.test_acc_list[t].append(self.test_acc[t])
        else:
            data=self.ol()
            if data=='end':
                return
            self.total_epoch+=1
            with tf.GradientTape() as tape:
                output=self.nn.fp(data[0])
                train_loss=self.nn.loss(output,data[1])
            if self.optf!=True:
                self.apply_gradient(tape,self.opt,train_loss,self.nn.param)
            else:
                gradient=tape.gradient(train_loss,self.nn.param)
                self.sopt(gradient,self.nn.param)
            if self.total_epoch==1:
                loss=train_loss.numpy()
            self.nn.train_loss=loss.astype(np.float32)
            if self.eflag==True:
                self.nn.epochcount+=1
        return
    
    
    def train_(self,data_batch=None,labels_batch=None,batches=None,batch=None,test_batch=None,index1=None,index2=None,j=None,t=None):
        if batch!=None:
            if type(self.train_data)==list:
                for i in range(len(self.train_data)):
                    if batch!=1:
                        data_batch[i]=self.train_data[i][index1:index2]
                    else:
                        data_batch[i]=self.train_data[i][j]
            else:
                if batch!=1:
                    data_batch=self.train_data[index1:index2]
                else:
                    data_batch=self.train_data[j]
            if type(self.train_labels)==list:
                for i in range(len(self.train_data)):
                    if batch!=1:
                        labels_batch[i]=self.train_labels[i][index1:index2]
                    else:
                        labels_batch[i]=self.train_labels[i][j]
            else:
                if batch!=1:
                    labels_batch=self.train_labels[index1:index2]
                else:
                    labels_batch=self.train_labels[j]
            if self.PO==1:
                with tf.GradientTape() as tape:
                    self.output=self.nn.fp(data_batch)
                    self.batch_loss=self.nn.loss(self.output,labels_batch)
                self.gradient=tape.gradient(self.batch_loss,self.nn.param)
                if self.optf!=True:
                    self.opt.apply_gradients(zip(self.gradient,self.nn.param))
                else:
                    self.sopt(self.gradient,self.nn.param,t)
                if self.total_epoch==1:
                    self.batch_loss=self.batch_loss.numpy()
                if self.acc_flag1==1:
                    self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                    self.batch_acc=self.batch_acc.numpy()
                if self.bflag==True:
                    self.nn.batchcount=j
            else:
                self.thread_lock.acquire()
                self.param=self.nn.param
                with tf.GradientTape() as tape:
                    self.output=self.nn.fp(data_batch)
                    self.batch_loss=self.nn.loss(self.output,labels_batch)
                self.gradient=tape.gradient(self.batch_loss,self.param)
                self.thread_lock.release()
                self.thread_lock.acquire()
                if self.optf!=True:
                    self.opt.apply_gradients(zip(self.gradient,self.nn.param))
                else:
                    self.sopt(self.gradient,self.nn.param,t)
                if self.total_epoch==1:
                    self.batch_loss=self.batch_loss.numpy()
                if self.acc_flag1==1:
                    self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                    self.batch_acc=self.batch_acc.numpy()
                if self.bflag==True:
                    self.nn.batchcount=j
                if self.acc_flag1==1 and batch!=None:
                    return self.batch_loss,self.batch_acc
                elif batch!=None:
                    return self.batch_loss
                self.thread_lock.release()
            if index1==batches*batch:
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        data_batch[i]=tf.concat([self.train_data[i][index1:],self.train_data[i][:index2]])
                else:
                    data_batch=tf.concat([self.train_data[index1:],self.train_data[:index2]])
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=tf.concat([self.train_labels[i][index1:],self.train_labels[i][:index2]])
                else:
                    labels_batch=tf.concat([self.train_labels[index1:],self.train_labels[:index2]])
                if self.PO==1:
                    with tf.GradientTape() as tape:
                        self.output=self.nn.fp(data_batch)
                        self.batch_loss=self.nn.loss(self.output,labels_batch)
                    self.gradient=tape.gradient(self.batch_loss,self.nn.param)
                    if self.optf!=True:
                        self.opt.apply_gradients(zip(self.gradient,self.nn.param))
                    else:
                        self.sopt(self.gradient,self.param,t)
                    if self.total_epoch==1:
                        self.batch_loss=self.batch_loss.numpy()
                    if self.acc_flag1==1:
                        self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                        self.batch_acc=self.batch_acc.numpy()
                    if self.bflag==True:
                        self.nn.batchcount+=1
                else:
                    self.thread_lock.acquire()
                    self.param=self.nn.param
                    with tf.GradientTape() as tape:
                        self.output=self.nn.fp(data_batch)
                        self.batch_loss=self.nn.loss(self.output,labels_batch)
                    self.gradient=tape.gradient(self.batch_loss,self.param)
                    self.thread_lock.release()
                    self.thread_lock.acquire()
                    if self.optf!=True:
                        self.opt.apply_gradients(zip(self.gradient,self.nn.param))
                    else:
                        self.sopt(self.gradient,self.nn.param,t)
                    if self.total_epoch==1:
                        self.batch_loss=self.batch_loss.numpy()
                    if self.acc_flag1==1:
                        self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                        self.batch_acc=self.batch_acc.numpy()
                    if self.bflag==True:
                        self.nn.batchcount+=1
                    if self.acc_flag1==1 and batch!=None:
                        return self.batch_loss,self.batch_acc
                    elif batch!=None:
                        return self.batch_loss
                    self.thread_lock.release()
        else:
            if self.PO==1:
                with tf.GradientTape() as tape:
                    self.output=self.nn.fp(self.train_data)
                    self._train_loss=self.nn.loss(self.output,self.train_labels)
                self.gradient=tape.gradient(self._train_loss,self.nn.param)
                if self.optf!=True:
                    self.opt.apply_gradients(zip(self.gradient,self.nn.param))
                else:
                    self.sopt(self.gradient,self.nn.param)
                if self.total_epoch==1:
                    self.loss=self._train_loss.numpy()
                self.train_loss_list.append(self.loss.astype(np.float32))
                self.train_loss=self.loss
                self.train_loss=self.train_loss.astype(np.float32)
                if self.acc_flag1==1:
                    self.acc=self.nn.accuracy(self.output,self.train_labels)
                    self.acc=self.acc.numpy()
                    self.train_acc_list.append(self.acc.astype(np.float32))
                    self.train_acc=self.acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if self.test==True:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    if self.acc_flag1==1:
                        self.test_acc_list.append(self.test_acc)
            else:
                self.thread_lock.acquire()
                self.param=self.nn.param
                with tf.GradientTape() as tape:
                    self.output=self.nn.fp(self.train_data)
                    self._train_loss=self.nn.loss(self.output,self.train_labels)
                self.gradient=tape.gradient(self._train_loss,self.param)
                self.thread_lock.release()
                self.thread_lock.acquire()
                if self.optf!=True:
                    self.opt.apply_gradients(zip(self.gradient,self.nn.param))
                else:
                    self.sopt(self.gradient,self.nn.param,t)
                if self.total_epoch==1:
                    self.loss=self._train_loss.numpy()
                self.train_loss_list.append(self.loss.astype(np.float32))
                self.train_loss=self.loss
                self.train_loss=self.train_loss.astype(np.float32)
                if self.acc_flag1==1:
                    self.acc=self.nn.accuracy(self.output,self.train_labels)
                    self.acc=self.acc.numpy()
                    self.train_acc_list.append(self.acc.astype(np.float32))
                    self.train_acc=self.acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if self.test==True:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    if self.acc_flag1==1:
                        self.test_acc_list.append(self.test_acc)
                self.thread_lock.release()
            return
    
    
    def _train_(self,batch=None,data_batch=None,labels_batch=None,test_batch=None,t=None):
        total_loss=0
        total_acc=0
        batches=int((self.shape0-self.shape0%batch)/batch)
        for j in range(batches):
            index1=j*batch
            index2=(j+1)*batch
            if self.acc_flag1==1:
                batch_loss,batch_acc=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
                total_loss+=batch_loss
                total_acc+=batch_acc
            else:
                batch_loss=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
                total_loss+=batch_loss
        if self.shape0%batch!=0:
            batches+=1
            index1=batches*batch
            index2=batch-(self.shape0-batches*batch)
            self.train_(batch,test_batch,index1,index2,t)
        loss=total_loss/batches
        if self.acc_flag1==1:
            train_acc=total_acc/batches
        self.train_loss_list.append(loss.astype(np.float32))
        self.train_loss=loss
        self.train_loss=self.train_loss.astype(np.float32)
        if self.acc_flag1==1:
            self.train_acc_list.append(train_acc.astype(np.float32))
            self.train_acc=train_acc
            self.train_acc=self.train_acc.astype(np.float32)
        if self.test==True:
            self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
            self.test_loss_list.append(self.test_loss)
        if self.acc_flag1==1:
            self.test_acc_list.append(self.test_acc)
        return
    
    
    def train(self,batch=None,epoch=None,test_batch=None,nn_path=None,one=True,p=None,s=None):
        self.batch=batch
        self.epoch=0
        try:
            if self.nn.km==0:
                self.nn.km=1
        except AttributeError:
            pass
        if self.flag==None:
            self.flag=True
        if p==None and s==None:
            self.p=9
            self.s=2
        elif p!=None:
            self.p=p-1
            self.s=2
        elif s!=None:
            self.p=9
            self.s=s
        else:
            self.p=p-1
            self.s=s
        if self.optf!=True:
            self.opt=self.nn.opt
        else:
            self.sopt=self.nn.opt
        if type(self.train_data)==list:
            data_batch=[x for x in range(len(self.train_data))]
        if type(self.train_labels)==list:
            labels_batch=[x for x in range(len(self.train_labels))]
        if epoch!=None:
            for i in range(epoch):
                t1=time.time()
                if self.eflag==True:
                    if self.thread==None:
                        self.nn.epochcount+=1
                    else:
                        self.nn.epochcount[self.t[-1]]+=1
                if self.thread==None:
                    self._train(batch,test_batch,data_batch,labels_batch)
                else:
                    t=self.t.pop()
                    if self.PO==1:
                        self.thread_lock.acquire()
                        self._train_(batch,data_batch,labels_batch,test_batch,t)
                        self.thread_lock.release()
                    elif self.PO!=None:
                        self._train_(batch,data_batch,labels_batch,test_batch,t)
                    else:
                        self._train(batch,test_batch,data_batch,labels_batch,t)
                if self.thread==None:
                    self.epoch+=1
                    self.total_epoch+=1
                else:
                    self.epoch[t]+=1
                    self.total_epoch[t]+=1
                if self.thread==None:
                    if epoch%10!=0:
                        d=epoch-epoch%self.p
                        d=int(d/self.p)
                    else:
                        d=epoch/(self.p+1)
                        d=int(d)
                    if d==0:
                        d=1
                    e=d*self.s
                    if i%d==0:
                        if self.flag==None:
                            if self.test==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                        else:
                            if self.test==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch+i+1,self.train_loss,self.test_loss))
                        if nn_path!=None and i%e==0:
                            self.save(nn_path,i,one)
                t2=time.time()
                if self.thread==None:
                    self.time+=(t2-t1)
                else:
                    self.time[t]+=(t2-t1)
                if self.end()==True:
                    break
        elif self.ol==None:
            i=0
            while True:
                t1=time.time()
                i+=1
                if self.thread==None:
                    self._train(test_batch=test_batch)
                else:
                    t=self.t.pop()
                    if self.PO==1:
                        self.thread_lock.acquire()
                        self._train_(test_batch=test_batch,t=t)
                        self.thread_lock.release()
                    elif self.PO!=None:
                        self._train_(test_batch=test_batch,t=t)
                    else:
                        self._train(test_batch=test_batch,t=t)
                if self.thread==None:
                    self.epoch+=1
                    self.total_epoch+=1
                else:
                    self.epoch[t]+=1
                    self.total_epoch[t]+=1
                if self.thread==None:
                    if epoch%10!=0:
                        d=epoch-epoch%self.p
                        d=int(d/self.p)
                    else:
                        d=epoch/(self.p+1)
                        d=int(d)
                    if d==0:
                        d=1
                    e=d*self.s
                    if i%d==0:
                        if self.flag==None:
                            if self.test==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                        else:
                            if self.test==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch+i+1,self.train_loss,self.test_loss))
                        if nn_path!=None and i%e==0:
                            self.save(nn_path,i,one)
                if self.eflag==True:
                    if self.thread==None:
                        self.nn.epochcount+=1
                    else:
                        self.nn.epochcount[t]+=1
                t2=time.time()
                if self.thread==None:
                    self.time+=(t2-t1)
                else:
                    self.time[t]+=(t2-t1)
                if self.end()==True:
                    break
        else:
            while True:
                self._train()
                if nn_path!=None:
                    self.save(nn_path)
                if self.eflag==True:
                    self.nn.epochcount+=1
        if nn_path!=None:
            self.save(nn_path)
        if self.thread==None:
            self.time=self.time-int(self.time)
            if self.time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            self.total_time+=self.time
        else:
            self.time[t]=self.time[t]-int(self.time[t])
            if self.time[t]<0.5:
                self.time[t]=int(self.time[t])
            else:
                self.time[t]=int(self.time[t])+1
            self.total_time[t]+=self.time[t]
        if self.thread==None:
            print()
            if self.test==False:
                print('last loss:{0:.6f}'.format(self.train_loss))
            else:
                print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))
            if self.acc_flag1==1:
                if self.acc_flag2=='%':
                    if self.test==False:
                        print('accuracy:{0:.1f}'.format(self.train_acc*100))
                    else:
                        print('accuracy:{0:.1f},test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))
                else:
                    if self.test==False:
                        print('accuracy:{0:.6f}'.format(self.train_acc))
                    else:
                        print('accuracy:{0:.6f},test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))   
            print('time:{0}s'.format(self.time))
        if self.thread==None:
            try:
                if self.nn.km==1:
                    self.nn.km=0
            except AttributeError:
                pass
        return
    
    
    def test(self,test_data,test_labels,batch=None,t=None):
        if type(test_data)==list:
            data_batch=[x for x in range(len(test_data))]
        if type(test_labels)==list:
            labels_batch=[x for x in range(len(test_labels))]
        if batch!=None:
            total_loss=0
            total_acc=0
            if type(test_data)==list:
                batches=int((test_data[0].shape[0]-test_data[0].shape[0]%batch)/batch)
                shape0=test_data[0].shape[0]
            else:
                batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
                shape0=test_data.shape[0]
            for j in range(batches):
                index1=j*batch
                index2=(j+1)*batch
                if type(test_data)==list:
                    for i in range(len(test_data)):
                        data_batch[i]=test_data[i][index1:index2]
                else:
                    data_batch=test_data[index1:index2]
                if type(test_labels)==list:
                    for i in range(len(test_labels)):
                        labels_batch[i]=test_labels[i][index1:index2]
                else:
                    labels_batch=test_labels[index1:index2]
                if self.thread==None:
                    output=self.nn.fp(data_batch)
                else:
                    output=self.nn.fp(data_batch,t)
                batch_loss=self.nn.loss(output,labels_batch)
                total_loss+=batch_loss.numpy()
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    total_acc+=batch_acc.numpy()
            if shape0%batch!=0:
                batches+=1
                index1=batches*batch
                index2=batch-(shape0-batches*batch)
                if type(test_data)==list:
                    for i in range(len(test_data)):
                        if type(test_data)==np.ndarray:
                            data_batch[i]=np.concatenate(test_data[i][index1:],test_data[i][:index2])
                        else:
                            data_batch[i]=tf.concat(test_data[i][index1:],test_data[i][:index2])
                else:
                    if type(test_data)==np.ndarray:
                        data_batch=np.concatenate(test_data[index1:],test_data[:index2])
                    else:
                        data_batch=tf.concat(test_data[index1:],test_data[:index2])
                if type(self.test_labels)==list:
                    for i in range(len(test_labels)):
                        if type(test_labels)==np.ndarray:
                            labels_batch[i]=np.concatenate(test_labels[i][index1:],test_labels[i][:index2])
                        else:
                            labels_batch[i]=tf.concat(test_labels[i][index1:],test_labels[i][:index2])
                else:
                    if type(test_labels)==np.ndarray:
                        labels_batch=np.concatenate(test_labels[index1:],test_labels[:index2])
                    else:
                        labels_batch=tf.concat(test_labels[index1:],test_labels[:index2])
                if self.thread==None:
                    output=self.nn.fp(data_batch)
                else:
                    output=self.nn.fp(data_batch,t)
                batch_loss=self.nn.loss(output,labels_batch)
                total_loss+=batch_loss.numpy()
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    total_acc+=batch_acc.numpy()
            test_loss=total_loss/batches
            test_loss=test_loss
            test_loss=test_loss.astype(np.float32)
            if self.acc_flag1==1:
                test_acc=total_acc/batches
                test_acc=test_acc
                test_acc=test_acc.astype(np.float32)
        else:
            if self.thread==None:
                output=self.nn.fp(test_data)
            else:
                output=self.nn.fp(test_data,t)
            test_loss=self.nn.loss(output,test_labels)
            if self.acc_flag1==1:
                test_acc=self.nn.accuracy(output,test_labels)
                test_loss=test_loss.numpy().astype(np.float32)
                test_acc=test_acc.numpy().astype(np.float32)
        if self.thread==None:
            print('test loss:{0:.6f}'.format(test_loss))
            if self.acc_flag1==1:
                if self.acc_flag2=='%':
                    print('accuracy:{0:.1f}'.format(test_acc*100))
                else:
                    print('accuracy:{0:.6f}'.format(test_acc))
                if self.acc_flag2=='%':
                    return test_loss,test_acc*100
                else:
                    return test_loss,test_acc
            else:
                return test_loss
    
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.total_epoch))
        print()
        print('learning rate:{0}'.format(self.nn.lr))
        print()
        print('time:{0:.3f}s'.format(self.total_time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0:.6f}'.format(self.train_loss))
        if self.acc_flag2=='%':
            print('train acc:{0:.1f}'.format(self.train_acc*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc))       
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        if self.acc_flag2=='%':
            print('test acc:{0:.1f}'.format(self.test_acc*100))
        else:
            print('test acc:{0:.6f}'.format(self.test_acc))      
        return
		
    
    def info(self):
        self.train_info()
        if self.test==True:
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
        if self.acc_flag2=='%':
            print('train acc:{0:.1f}'.format(self.train_acc*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc))    
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
        if self.acc_flag2=='%':
            print('test acc:{0:.1f}'.format(self.test_acc*100))
        else:
            print('test acc:{0:.6f}'.format(self.test_acc))  
        return 
    
    
    def comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.train_loss_list,'b-',label='train loss')
        if self.test==True:
            plt.plot(np.arange(self.total_epoch),self.test_loss_list,'r-',label='test loss')
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.train_acc_list,'b-',label='train acc')
        if self.test==True:
            plt.plot(np.arange(self.total_epoch),self.test_acc_list,'r-',label='test acc')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        print('train loss:{0}'.format(self.train_loss))
        if self.acc_flag2=='%':
            print('train acc:{0:.1f}'.format(self.train_acc*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc))     
        if self.test==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            if self.acc_flag2=='%':
                print('test acc:{0:.1f}'.format(self.test_acc*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc)) 
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump(self.nn.param,parameter_file)
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
        pickle.dump(self.nn.param,parameter_file)
        self.nn.param=None
        pickle.dump(self.nn,output_file)
        pickle.dump(self.ol,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.end_acc,output_file)
        pickle.dump(self.end_test_loss,output_file)
        pickle.dump(self.end_test_acc,output_file)
        if self.eflag==True:
            pickle.dump(self.eflag,output_file)
        if self.bflag==True:
            pickle.dump(self.bflag,output_file)
        pickle.dump(self.optf,output_file)
        pickle.dump(self.acc_flag1,output_file)
        pickle.dump(self.acc_flag2,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.s,output_file)
        pickle.dump(self.flag,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_acc,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_acc_list,output_file)
        pickle.dump(self.test,output_file)
        if self.test==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_acc,output_file)
            pickle.dump(self.test_loss_list,output_file)
            pickle.dump(self.test_acc_list,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        parameter_file.close()
        return
    
	
    def restore(self,s_path,p_path):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        param=pickle.load(parameter_file)
        self.nn=pickle.load(input_file)
        self.nn.param=param
        param=None
        if self.nn.optf!=True:
            self.opt=self.nn.opt
        else:
            self.sopt=self.nn.opt
        self.ol=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.end_acc=pickle.load(input_file)
        self.end_test_loss=pickle.load(input_file)
        self.end_test_acc=pickle.load(input_file)
        if self.eflag==True:
            self.eflag=pickle.load(input_file)
        if self.bflag==True:
            self.bflag=pickle.load(input_file)
        self.optf=pickle.load(input_file)
        self.acc_flag1=pickle.load(input_file)
        self.acc_flag2=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.s=pickle.load(input_file)
        self.flag=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_acc=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_acc_list=pickle.load(input_file)
        self.test=pickle.load(input_file)
        if self.test==True:
            self.test_loss=pickle.load(input_file)
            self.test_acc=pickle.load(input_file)
            self.test_loss_list=pickle.load(input_file)
            self.test_acc_list=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        parameter_file.close()
        return
