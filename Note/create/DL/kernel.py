import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time


class kernel:
    def __init__(self,nn=None):
        if nn!=None:
            self.nn=nn
            try:
                self.nn.km=1
            except AttributeError:
                pass
        self.PO=None
        self.thread_lock=None
        self.thread=None
        self.ol=None
        self.batch=None
        self.epoch=0
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.acc_flag1=None
        self.acc_flag2='%'
        self.train_flag=None
        self.train_counter=0
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.test_flag=False
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
            self.test_flag=True
        if type(self.train_data)==list:
            self.shape0=train_data[0].shape[0]
        else:
            self.shape0=train_data.shape[0]
        if self.thread!=None:
            self.t=-np.arange(-(self.thread-1),1)
            self.t=list(self.t)
            try:
                self.nn.ec=np.zeros(self.thread)
            except AttributeError:
                pass
            try:
                self.nn.bc=np.zeros(self.thread)
            except AttributeError:
                pass
            if self.PO==None:
                self.train_loss=np.zeros(self.thread)
                self.train_acc=np.zeros(self.thread)
                self.train_loss_list=[[] for _ in range(self.thread)]
                self.train_acc_list=[[] for _ in range(self.thread)]
            if test_data!=None:
                if self.PO==None:
                    self.test_loss=np.zeros(self.thread)
                    self.test_acc=np.zeros(self.thread)
                    self.test_loss_list=[[] for _ in range(self.thread)]
                    self.test_acc_list=[[] for _ in range(self.thread)]
            self.epoch=np.zeros(self.thread)
            self.total_epoch=np.zeros(self.thread)
            self.time=np.zeros(self.thread)
            self.total_time=np.zeros(self.thread)
        return
    
    
    def init(self):
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        self.test_loss_list.clear()
        self.test_acc_list.clear()
        self.test_flag=False
        self.train_counter=0
        self.epoch=0
        self.total_epoch=0
        self.time=0
        self.total_time=0
        return
    
    
    def add_threads(self,thread):
        t=-np.arange(-thread+1,1)+self.thread
        self.t=t.extend(self.t)
        self.thread+=thread
        try:
            self.nn.ec=np.concatenate((self.nn.ec,np.zeros(thread)))
        except AttributeError:
            pass
        try:
            self.nn.bc=np.concatenate((self.nn.bc,np.zeros(thread)))
        except AttributeError:
            pass
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
    
    
    def loss_acc(self,output=None,labels_batch=None,loss=None,test_batch=None,total_loss=None,total_acc=None,t=None):
        if self.batch!=None:
            total_loss+=loss
            if self.acc_flag1==1:
                batch_acc=self.nn.accuracy(output,labels_batch)
                total_acc+=batch_acc
            return total_loss,total_acc
        elif self.ol==None:
            loss=loss.numpy()
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
            if self.test_flag==True:
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
            return
    
    
    def _train(self,batch=None,data_batch=None,labels_batch=None,test_batch=None,t=None):
        if self.end_loss!=None or self.end_acc!=None or self.end_test_loss!=None or self.end_test_acc!=None:
            self._param=self.nn.param
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
                try:
                    if self.nn.gradient!=None:
                        pass
                    if self.thread==None:
                        output=self.nn.fp(data_batch)
                    else:
                        output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                except AttributeError:
                    with tf.GradientTape() as tape:
                        if self.thread==None:
                            output=self.nn.fp(data_batch)
                        else:
                            output=self.nn.fp(data_batch,t)
                        batch_loss=self.nn.loss(output,labels_batch)
                try:
                    if self.thread==None:
                        if self.nn.opt!=None:
                            pass
                        gradient=tape.gradient(batch_loss,self.nn.param)
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                    else:
                        if self.nn.opt!=None:
                            pass
                        gradient=tape.gradient(batch_loss,self.nn.param[t])
                        self.nn.opt[t].apply_gradients(zip(gradient,self.nn.param[t]))
                except AttributeError:
                    if self.thread==None:
                        gradient=self.nn.gradient(batch_loss,self.nn.param)
                        self.nn.oopt(gradient,self.nn.param)
                    else:
                        gradient=self.nn.gradient(batch_loss,self.nn.param[t])
                        self.nn.oopt(gradient,self.nn.param,t)
                total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc,t=t)
                if self.thread==None:
                    try:
                        self.nn.bc=j
                    except AttributeError:
                        pass
                else:
                    try:
                        self.nn.bc[t]=j
                    except AttributeError:
                        pass
            if self.shape0%batch!=0:
                batches+=1
                index1=batches*batch
                index2=batch-(self.shape0-batches*batch)
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        data_batch[i]=tf.concat([self.train_data[i][index1:],self.train_data[i][:index2]],0)
                else:
                    data_batch=tf.concat([self.train_data[index1:],self.train_data[:index2]],0)
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=tf.concat([self.train_labels[i][index1:],self.train_labels[i][:index2]],0)
                else:
                    labels_batch=tf.concat([self.train_labels[index1:],self.train_labels[:index2]],0)
                try:
                    if self.nn.gradient!=None:
                        pass
                    if self.thread==None:
                        output=self.nn.fp(data_batch)
                    else:
                        output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                except AttributeError:
                    with tf.GradientTape() as tape:
                        if self.thread==None:
                            output=self.nn.fp(data_batch)
                        else:
                            output=self.nn.fp(data_batch,t)
                        batch_loss=self.nn.loss(output,labels_batch)
                try:
                    if self.thread==None:
                        if self.nn.opt!=None:
                            pass
                        gradient=tape.gradient(batch_loss,self.nn.param)
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                    else:
                        if self.nn.opt:
                            pass
                        gradient=tape.gradient(batch_loss,self.nn.param[t])
                        self.nn.opt[t].apply_gradients(zip(gradient,self.nn.param[t]))
                except AttributeError:
                    if self.thread==None:
                        gradient=self.nn.gradient(batch_loss,self.nn.param)
                        self.nn.oopt(gradient,self.param)
                    else:
                        gradient=self.nn.gradient(batch_loss,self.nn.param[t])
                        self.nn.oopt(gradient,self.nn.param,t)
                total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc,t=t)
                if self.thread==None:
                    try:
                        self.nn.bc+=1
                    except AttributeError:
                        pass
                else:
                    try:
                        self.nn.bc[t]+=1
                    except AttributeError:
                        pass
            loss=total_loss.numpy()/batches
            if self.acc_flag1==1:
                train_acc=total_acc.numpy()/batches
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
            if self.test_flag==True:
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
            try:
                if self.nn.gradient!=None:
                    if self.thread==None:
                        output=self.nn.fp(self.train_data)
                    else:
                        output=self.nn.fp(data_batch,t)
                    train_loss=self.nn.loss(output,self.train_labels)
            except AttributeError:
                with tf.GradientTape() as tape:
                    if self.thread==None:
                        output=self.nn.fp(self.train_data)
                    else:
                        output=self.nn.fp(data_batch,t)
                    train_loss=self.nn.loss(output,self.train_labels)
            try:
                if self.thread==None:
                    if self.nn.opt!=None:
                            pass
                    gradient=tape.gradient(train_loss,self.nn.param)
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                else:
                    if self.nn.opt:
                            pass
                    gradient=tape.gradient(train_loss,self.nn.param[t])
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param[t]))
            except AttributeError:
                if self.thread==None:
                    gradient=self.nn.gradient(train_loss,self.nn.param)
                    self.nn.oopt(gradient,self.nn.param)
                else:
                    gradient=self.nn.gradient(batch_loss,self.nn.param[t])
                    self.nn.oopt(gradient,self.nn.param,t)
            self.loss_acc(output=output,labels_batch=labels_batch,loss=train_loss,test_batch=test_batch,total_loss=total_loss,total_acc=total_acc,t=t)
        else:
            data=self.ol()
            try:
                if self.nn.gradient!=None:
                    output=self.nn.fp(data[0])
                    train_loss=self.nn.loss(output,data[1])
            except AttributeError:
                with tf.GradientTape() as tape:
                    output=self.nn.fp(data[0])
                    train_loss=self.nn.loss(output,data[1])
            if self.thread_lock!=None:
                try:
                    if self.nn.opt!=None:
                        pass
                    if self.PO==1:
                        gradient=tape.gradient(train_loss,self.nn.param)
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                    else:
                        self.thread_lock[0].acquire()
                        self.param=self.nn.param
                        self.gradient=tape.gradient(train_loss,self.param)
                        self.thread_lock[0].release()
                        self.thread_lock[1].acquire()
                        self.nn.opt.apply_gradients(zip(self.gradient,self.nn.param))
                        self.thread_lock[1].release()
                except AttributeError:
                    if self.PO==1:
                        self.gradient=self.nn.gradient(train_loss,self.nn.param)
                        self.nn.oopt(self.gradient,self.nn.param)
                    else:
                        self.thread_lock[0].acquire()
                        self.gradient=self.nn.gradient(train_loss,self.nn.param)
                        self.thread_lock[0].release()
                        self.thread_lock[1].acquire()
                        self.nn.oopt(self.gradient,self.nn.param)
                        self.thread_lock[1].release()
            else:
                try:
                    if self.nn.opt!=None:
                        pass
                    self.apply_gradient(tape,self.nn.opt,train_loss,self.nn.param)
                except AttributeError:
                    gradient=self.nn.gradient(train_loss,self.nn.param)
                    self.nn.oopt(gradient,self.nn.param)
            loss=train_loss.numpy()
            if self.thread_lock!=None:
                self.thread_lock[2].acquire()
                self.nn.train_loss=loss.astype(np.float32)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                self.total_epoch+=1
                self.thread_lock[2].release()
            else:
                self.nn.train_loss=loss.astype(np.float32)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                self.total_epoch+=1
        return
    
    
    def train_(self,data_batch=None,labels_batch=None,batch=None,batches=None,test_batch=None,index1=None,index2=None,j=None,t=None):
        if self.end_loss!=None or self.end_acc!=None or self.end_test_loss!=None or self.end_test_acc!=None:
            self._param=self.nn.param
        if batch!=None:
            if index1==batches*batch:
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        data_batch[i]=tf.concat([self.train_data[i][index1:],self.train_data[i][:index2]],0)
                else:
                    data_batch=tf.concat([self.train_data[index1:],self.train_data[:index2]],0)
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=tf.concat([self.train_labels[i][index1:],self.train_labels[i][:index2]],0)
                else:
                    labels_batch=tf.concat([self.train_labels[index1:],self.train_labels[:index2]],0)
                if self.PO==1:
                    with tf.GradientTape() as tape:
                        self.output=self.nn.fp(data_batch)
                        self.batch_loss=self.nn.loss(self.output,labels_batch)
                    try:
                        if self.nn.opt!=None:
                            pass
                        gradient=tape.gradient(self.batch_loss,self.nn.param)
                    except AttributeError:
                        gradient=self.nn.gradient(tape,self.batch_loss,self.nn.param)
                    try:
                        if self.nn.opt!=None:
                            pass
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                    except AttributeError:
                        self.nn.oopt(gradient,self.param,t)
                    if self.acc_flag1==1:
                        self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                    try:
                        self.nn.bc=j
                    except AttributeError:
                        pass
                else:
                    self.thread_lock[0].acquire()
                    with tf.GradientTape() as tape:
                        self.output=self.nn.fp(data_batch)
                        self.batch_loss=self.nn.loss(self.output,labels_batch)
                    try:
                        if self.nn.opt!=None:
                            pass
                        self.gradient=tape.gradient(self.batch_loss,self.nn.param)
                    except AttributeError:
                        self.gradient=self.nn.gradient(tape,self.batch_loss,self.nn.param)
                    self.thread_lock[0].release()
                    self.thread_lock[1].acquire()
                    try:
                        if self.nn.opt!=None:
                            pass
                        self.nn.opt.apply_gradients(zip(self.gradient,self.nn.param))
                    except AttributeError:
                        self.nn.oopt(self.gradient,self.nn.param,t)
                    if self.acc_flag1==1:
                        self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                    try:
                        self.nn.bc+=1
                    except AttributeError:
                        pass
                    if self.acc_flag1==1:
                        return self.batch_loss,self.batch_acc
                    else:
                        return self.batch_loss
                    self.thread_lock[1].release()
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
                try:
                    if self.nn.opt!=None:
                        pass
                    gradient=tape.gradient(self.batch_loss,self.nn.param)
                except AttributeError:
                    gradient=self.nn.gradient(tape,self.batch_loss,self.nn.param)
                try:
                    if self.nn.opt!=None:
                        pass
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                    self.nn.oopt(gradient,self.nn.param,t)
                if self.acc_flag1==1:
                    self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                try:
                    self.nn.bc=j
                except AttributeError:
                    pass
            else:
                self.thread_lock[0].acquire()
                with tf.GradientTape() as tape:
                    self.output=self.nn.fp(data_batch)
                    self.batch_loss=self.nn.loss(self.output,labels_batch)
                try:
                    if self.nn.opt!=None:
                        pass
                    self.gradient=tape.gradient(self.batch_loss,self.nn.param)
                except AttributeError:
                    self.gradient=self.nn.gradient(tape,self.batch_loss,self.nn.param)
                self.thread_lock[0].release()
                self.thread_lock[1].acquire()
                try:
                    if self.nn.opt!=None:
                        pass
                    self.nn.opt.apply_gradients(zip(self.gradient,self.nn.param))
                except AttributeError:
                    self.nn.oopt(self.gradient,self.nn.param,t)
                if self.acc_flag1==1:
                    self.batch_acc=self.nn.accuracy(self.output,labels_batch)
                try:
                    self.nn.bc=j
                except AttributeError:
                    pass
                if self.acc_flag1==1:
                    return self.batch_loss,self.batch_acc
                else:
                    return self.batch_loss
                self.thread_lock[1].release()
        else:
            if self.PO==1:
                with tf.GradientTape() as tape:
                    self.output=self.nn.fp(self.train_data)
                    self.train_loss=self.nn.loss(self.output,self.train_labels)
                try:
                    if self.nn.opt!=None:
                        pass
                    gradient=tape.gradient(self.train_loss,self.nn.param)
                except AttributeError:
                    gradient=self.nn.gradient(tape,self.train_loss,self.nn.param)
                try:
                    if self.nn.opt!=None:
                        pass
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                    self.nn.oopt(gradient,self.nn.param)
                self.loss=self.train_loss.numpy()
                self.train_loss_list.append(self.loss.astype(np.float32))
                self.train_loss=self.loss
                self.train_loss=self.train_loss.astype(np.float32)
                if self.acc_flag1==1:
                    self.acc=self.nn.accuracy(self.output,self.train_labels)
                    self.acc=self.acc.numpy()
                    self.train_acc_list.append(self.acc.astype(np.float32))
                    self.train_acc=self.acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if self.test_flag==True:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    if self.acc_flag1==1:
                        self.test_acc_list.append(self.test_acc)
            else:
                self.thread_lock[0].acquire()
                with tf.GradientTape() as tape:
                    self.output=self.nn.fp(self.train_data)
                    self.train_loss=self.nn.loss(self.output,self.train_labels)
                try:
                    if self.nn.opt!=None:
                        pass
                    self.gradient=tape.gradient(self.train_loss,self.nn.param)
                except AttributeError:
                    self.gradient=self.nn.gradient(tape,self.train_loss,self.nn.param)
                self.thread_lock[0].release()
                self.thread_lock[1].acquire()
                try:
                    if self.nn.opt!=None:
                        pass
                    self.nn.opt.apply_gradients(zip(self.gradient,self.nn.param))
                except AttributeError:
                    self.nn.oopt(self.gradient,self.nn.param,t)
                self.loss=self.train_loss.numpy()
                self.train_loss_list.append(self.loss.astype(np.float32))
                self.train_loss=self.loss
                self.train_loss=self.train_loss.astype(np.float32)
                if self.acc_flag1==1:
                    self.acc=self.nn.accuracy(self.output,self.train_labels)
                    self.acc=self.acc.numpy()
                    self.train_acc_list.append(self.acc.astype(np.float32))
                    self.train_acc=self.acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if self.test_flag==True:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    if self.acc_flag1==1:
                        self.test_acc_list.append(self.test_acc)
                self.thread_lock[1].release()
            return
    
    
    def _train_(self,batch=None,data_batch=None,labels_batch=None,test_batch=None,t=None):
        total_loss=0
        total_acc=0
        batches=int((self.shape0-self.shape0%batch)/batch)
        for j in range(batches):
            index1=j*batch
            index2=(j+1)*batch
            if self.acc_flag1==1:
                self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
                total_loss+=self.batch_loss
                total_acc+=self.batch_acc
            else:
                self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
                total_loss+=self.batch_loss
        if self.shape0%batch!=0:
            batches+=1
            index1=batches*batch
            index2=batch-(self.shape0-batches*batch)
            self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
            if self.acc_flag1==1:
                total_loss+=self.batch_loss
                total_acc+=self.batch_acc
            else:
                total_loss+=self.batch_loss
        loss=total_loss.numpy()/batches
        if self.acc_flag1==1:
            train_acc=total_acc.numpy()/batches
        self.train_loss_list.append(loss.astype(np.float32))
        self.train_loss=loss
        self.train_loss=self.train_loss.astype(np.float32)
        if self.acc_flag1==1:
            self.train_acc_list.append(train_acc.astype(np.float32))
            self.train_acc=train_acc
            self.train_acc=self.train_acc.astype(np.float32)
        if self.test_flag==True:
            self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
            self.test_loss_list.append(self.test_loss)
        if self.acc_flag1==1:
            self.test_acc_list.append(self.test_acc)
        return
    
    
    def train(self,batch=None,epoch=None,test_batch=None,save=None,one=True,p=None,s=None):
        self.train_flag=True
        self.batch=batch
        self.epoch=0
        self.train_counter+=1
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if s==None:
            self.s=1
            self.file_list=None
        else:
            self.s=s-1
            self.file_list=[]
        if type(self.train_data)==list:
            data_batch=[x for x in range(len(self.train_data))]
        else:
            data_batch=None
        if type(self.train_labels)==list:
            labels_batch=[x for x in range(len(self.train_labels))]
        else:
            labels_batch=None
        if epoch!=None:
            for i in range(epoch):
                t1=time.time()
                if self.thread==None:
                    self._train(batch,data_batch,labels_batch,test_batch)
                else:
                    t=self.t.pop()
                    if self.PO==1:
                        self.thread_lock.acquire()
                        self._train_(batch,data_batch,labels_batch,test_batch,t)
                        self.thread_lock.release()
                    elif self.PO!=None:
                        self._train_(batch,data_batch,labels_batch,test_batch,t)
                    else:
                        self._train(batch,data_batch,labels_batch,test_batch,t)
                if self.thread==None:
                    try:
                        self.nn.ec+=1
                    except AttributeError:
                        pass
                else:
                    try:
                        self.nn.ec[t]+=1
                    except AttributeError:
                        pass
                if type(self.total_epoch)!=list:
                    if self.thread_lock!=None:
                        if type(self.thread_lock)!=list:
                            self.thread_lock.acquire()
                            self.total_epoch+=1
                            self.thread_lock.release()
                        else:
                            self.thread_lock[2].acquire()
                            self.total_epoch+=1
                            self.thread_lock[2].release() 
                    else:
                        self.epoch+=1
                        self.total_epoch+=1
                else:
                    self.epoch[t]+=1
                    self.total_epoch[t]+=1
                if self.thread==None:
                    if epoch%10!=0:
                        p=epoch-epoch%self.p
                        p=int(p/self.p)
                        s=epoch-epoch%self.s
                        s=int(s/self.s)
                    else:
                        p=epoch/(self.p+1)
                        p=int(p)
                        s=epoch/(self.s+1)
                        s=int(s)
                    if p==0:
                        p=1
                    if s==0:
                        s=1
                    if i%p==0:
                        if self.train_counter==1:
                            if self.test_flag==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                        else:
                            if self.test_flag==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss,self.test_loss))
                    if save!=None and i%s==0:
                        self.save(self.total_epoch,one)
                t2=time.time()
                if self.thread==None:
                    self.time+=(t2-t1)
                else:
                    self.time[t]+=(t2-t1)
                if self.end_flag==True and self.end()==True:
                    self.nn.param=self._param
                    self._param=None
                    break
        elif self.ol==None:
            i=0
            while True:
                t1=time.time()
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
                i+=1
                if self.thread==None:
                    try:
                        self.nn.ec+=1
                    except AttributeError:
                        pass
                else:
                    try:
                        self.nn.ec[t]+=1
                    except AttributeError:
                        pass
                if type(self.total_epoch)!=list:
                    if self.thread_lock!=None:
                        if type(self.thread_lock)!=list:
                            self.thread_lock.acquire()
                            self.total_epoch+=1
                            self.thread_lock.release()
                        else:
                            self.thread_lock[2].acquire()
                            self.total_epoch+=1
                            self.thread_lock[2].release() 
                    else:
                        self.epoch+=1
                        self.total_epoch+=1
                else:
                    self.epoch[t]+=1
                    self.total_epoch[t]+=1
                if self.thread==None:
                    if epoch%10!=0:
                        p=epoch-epoch%self.p
                        p=int(p/self.p)
                        s=epoch-epoch%self.s
                        s=int(s/self.s)
                    else:
                        p=epoch/(self.p+1)
                        p=int(p)
                        s=epoch/(self.s+1)
                        s=int(s)
                    if p==0:
                        p=epoch
                    if s==0:
                        s=1
                    if i%p==0:
                        if self.train_counter==1:
                            if self.test_flag==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                        else:
                            if self.test_flag==False:
                                print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                            else:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss,self.test_loss))
                    if save!=None and i%s==0:
                        self.save(self.total_epoch,one)
                t2=time.time()
                if self.thread==None:
                    self.time+=(t2-t1)
                else:
                    self.time[t]+=(t2-t1)
                if self.end_flag==True and self.end()==True:
                    self.nn.param=self._param
                    self._param=None
                    break
        else:
            while True:
                self._train()
        if save!=None:
            self.save()
        if self.thread==None:
            self._time=self.time-int(self.time)
            if self._time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            self.total_time+=self.time
        elif type(self.total_time)==list:
            self.time[t]=self.time[t]-int(self.time[t])
            if self.time[t]<0.5:
                self.time[t]=int(self.time[t])
            else:
                self.time[t]=int(self.time[t])+1
            self.total_time[t]+=self.time[t]
        if self.thread==None:
            print()
            if self.test_flag==False:
                print('last loss:{0:.6f}'.format(self.train_loss))
            else:
                print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))
            if self.acc_flag1==1:
                if self.acc_flag2=='%':
                    if self.test_flag==False:
                        print('accuracy:{0:.1f}'.format(self.train_acc*100))
                    else:
                        print('accuracy:{0:.1f},test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))
                else:
                    if self.test_flag==False:
                        print('accuracy:{0:.6f}'.format(self.train_acc))
                    else:
                        print('accuracy:{0:.6f},test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))   
            print('time:{0}s'.format(self.time))
        self.train_flag=False
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
                total_loss+=batch_loss
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    total_acc+=batch_acc
            if shape0%batch!=0:
                batches+=1
                index1=batches*batch
                index2=batch-(shape0-batches*batch)
                if type(test_data)==list:
                    for i in range(len(test_data)):
                        data_batch[i]=tf.concat([test_data[i][index1:],test_data[i][:index2]],0)
                else:
                    data_batch=tf.concat([test_data[index1:],test_data[:index2]],0)
                if type(self.test_labels)==list:
                    for i in range(len(test_labels)):
                        labels_batch[i]=tf.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                else:
                    labels_batch=tf.concat([test_labels[index1:],test_labels[:index2]],0)
                if self.thread==None:
                    output=self.nn.fp(data_batch)
                else:
                    output=self.nn.fp(data_batch,t)
                batch_loss=self.nn.loss(output,labels_batch)
                total_loss+=batch_loss
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    total_acc+=batch_acc
            test_loss=total_loss.numpy()/batches
            test_loss=test_loss
            test_loss=test_loss.astype(np.float32)
            if self.acc_flag1==1:
                test_acc=total_acc.numpy()/batches
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
        try:
            print('learning rate:{0}'.format(self.nn.lr))
            print()
        except AttributeError:
            pass
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
        try:
            if self.nn.accuracy!=None:
                pass
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
        except AttributeError:
            pass
        return
    
    
    def test_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        try:
            if self.nn.accuracy!=None:
                pass
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
        except AttributeError:
            pass
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
        try:
            if self.nn.accuracy!=None:
                pass
            plt.figure(2)
            plt.plot(np.arange(self.total_epoch),self.train_acc_list,'b-',label='train acc')
            if self.test_flag==True:
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
        except AttributeError:
            pass
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            if self.acc_flag2=='%':
                print('test acc:{0:.1f}'.format(self.test_acc*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc)) 
        return
    
    
    def save_p(self):
        parameter_file=open('parameter.dat','wb')
        pickle.dump(self.nn.param,parameter_file)
        parameter_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if one==True:
            output_file=open('save.dat','wb')
        else:
            output_file=open('save-{0}.dat'.format(i),'wb')
            self.file_list.append(['save-{0}.dat'])
            if len(self.file_list)>self.s+1:
                os.remove(self.file_list[0][0])
                del self.file_list[0]
        try:
            if self.nn.opt:
                pass
            opt=self.nn.opt
            self.nn.opt=None
            pickle.dump(self.nn,output_file)
            self.nn.opt=opt
        except AttributeError:
            try:
                pickle.dump(self.nn,output_file)
            except:
                opt=self.nn.oopt
                self.nn.oopt=None
                pickle.dump(self.nn,output_file)
                self.nn.oopt=opt
        try:
            pickle.dump(opt.get_config(),output_file)
        except:
            pickle.dump(None,output_file)
        pickle.dump(self.ol,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.end_acc,output_file)
        pickle.dump(self.end_test_loss,output_file)
        pickle.dump(self.end_test_acc,output_file)
        pickle.dump(self.acc_flag1,output_file)
        pickle.dump(self.acc_flag2,output_file)
        pickle.dump(self.file_list,output_file)
        pickle.dump(self.train_counter,output_file)
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
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
	
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.config=pickle.load(input_file)
        self.ol=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.end_acc=pickle.load(input_file)
        self.end_test_loss=pickle.load(input_file)
        self.end_test_acc=pickle.load(input_file)
        self.acc_flag1=pickle.load(input_file)
        self.acc_flag2=pickle.load(input_file)
        self.file_list=pickle.load(input_file)
        self.train_counter=pickle.load(input_file)
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
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
