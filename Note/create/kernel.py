from tensorflow import function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.core=None
        self.PO=None
        self.thread_lock=None
        self.thread=None
        self.thread_counter=0
        self.ol=None
        self.suspend=False
        self.stop=None
        self.save_flag=None
        self.stop_flag=1
        self.training_flag=None
        self.save_epoch=None
        self.batch=None
        self.epoch=0
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.acc_flag='%'
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
            self.threadnum=np.arange(self.thread)
            self.threadnum=list(self.threadnum)
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
                self.epoch=np.zeros(self.thread)
                self.total_epoch=np.zeros(self.thread)
                self.time=np.zeros(self.thread)
                self.total_time=np.zeros(self.thread)
            if test_data!=None:
                if self.PO==None:
                    self.test_loss=np.zeros(self.thread)
                    self.test_acc=np.zeros(self.thread)
                    self.test_loss_list=[[] for _ in range(self.thread)]
                    self.test_acc_list=[[] for _ in range(self.thread)]
        return
    
    
    def init(self,param=None):
        if param!=None:
            self.nn.param=param
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
        threadnum=np.arange(thread)+self.thread
        self.threadnum=self.threadnum.extend(threadnum)
        self.thread+=thread
        try:
            self.nn.ec=np.concatenate((self.nn.ec,np.zeros(thread)))
        except AttributeError:
            pass
        try:
            self.nn.bc=np.concatenate((self.nn.bc,np.zeros(thread)))
        except AttributeError:
            pass
        if self.PO==None:
            self.train_loss=np.concatenate((self.train_loss,np.zeros(thread)))
            self.train_acc=np.concatenate((self.train_acc,np.zeros(thread)))
            self.train_loss_list.extend([[] for _ in range(thread)])
            self.train_acc_list.extend([[] for _ in range(thread)])
        if self.test_flag==True:
            if self.PO==None:
                self.test_loss=np.concatenate((self.test_loss,np.zeros(thread)))
                self.test_acc=np.concatenate((self.test_acc,np.zeros(thread)))
                self.test_loss_list.extend([[] for _ in range(thread)])
                self.test_acc_list.extend([[] for _ in range(thread)])
        self.epoch=np.concatenate((self.epoch,np.zeros(thread)))
        self.total_epoch=np.concatenate((self.total_epoch,np.zeros(thread)))
        self.time=np.concatenate((self.time,np.zeros(thread)))
        self.total_time=np.concatenate((self.total_time,np.zeros(thread)))
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
        if self.end_loss!=None and self.train_loss_list[-1]<=self.end_loss:
            return True
        elif self.end_acc!=None and self.train_acc_list[-1]>=self.end_acc:
            return True
        elif self.end_loss!=None and self.end_acc!=None and self.train_loss_list[-1]<=self.end_loss and self.train_acc_list[-1]>=self.end_acc:
            return True
        elif self.end_test_loss!=None and self.test_loss_list[-1]<=self.end_test_loss:
            return True
        elif self.end_test_acc!=None and self.test_acc_list[-1]>=self.end_test_acc:
            return True
        elif self.end_test_loss!=None and self.end_test_acc!=None and self.test_loss_list[-1]<=self.end_test_loss and self.test_acc_list[-1]>=self.end_test_acc:
            return True
    
    
    def loss_acc(self,output=None,labels_batch=None,loss=None,test_batch=None,total_loss=None,total_acc=None,t=None):
        if self.batch!=None:
            total_loss+=loss
            try:
                if self.nn.accuracy!=None:
                    pass
                batch_acc=self.nn.accuracy(output,labels_batch)
                total_acc+=batch_acc
            except AttributeError:
                pass
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
            try:
                if self.nn.accuracy!=None:
                    pass
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
            except AttributeError:
                pass
            if self.test_flag==True:
                if self.thread==None:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            pass
                        self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                else:
                    self.test_loss[t],self.test_acc[t]=self.test(self.test_data,self.test_labels,test_batch,t)
                    self.test_loss_list[t].append(self.test_loss[t])
                    try:
                        if self.nn.accuracy!=None:
                            pass
                        self.test_acc_list[t].append(self.test_acc[t])
                    except AttributeError:
                        pass
            return
    
    
    def data_func(self,data_batch=None,labels_batch=None,batch=None,index1=None,index2=None,j=None,flag=None):
        if flag==None:
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
        else:
            try:
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        data_batch[i]=self.core.concat([self.train_data[i][index1:],self.train_data[i][:index2]],0)
                else:
                    data_batch=self.core.concat([self.train_data[index1:],self.train_data[:index2]],0)
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=self.core.concat([self.train_labels[i][index1:],self.train_labels[i][:index2]],0)
                else:
                    labels_batch=self.core.concat([self.train_labels[index1:],self.train_labels[:index2]],0)
            except:
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        data_batch[i]=self.core.concat([self.train_data[i][index1:],self.train_data[i][:index2]],0)
                else:
                    data_batch=self.core.concat([self.train_data[index1:],self.train_data[:index2]],0)
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=self.core.concat([self.train_labels[i][index1:],self.train_labels[i][:index2]],0)
                else:
                    labels_batch=self.core.concat([self.train_labels[index1:],self.train_labels[:index2]],0)
        return data_batch,labels_batch
    
    
    @function
    def tf_opt(self,data,labels,t=None):
        try:
            if self.nn.GradientTape!=None:
                pass
            if self.thread==None:
                tape,output,loss=self.nn.GradientTape(data,labels)
            else:
                tape,output,loss=self.nn.GradientTape(data,labels,t)
        except AttributeError:
            with self.core.GradientTape(persistent=True) as tape:
                try:
                    if self.thread==None:
                        output=self.nn.fp(data)
                    else:
                        output=self.nn.fp(data,t)
                    loss=self.nn.loss(output,labels)
                except TypeError:
                    if self.thread==None:
                        output,loss=self.nn.fp(data,labels)
                    else:
                        output,loss=self.nn.fp(data,labels,t)
        if self.ol==None:
            try:
                if self.thread==None:
                    if self.nn.opt!=None:
                        pass
                    gradient=tape.gradient(loss,self.nn.param)
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                else:
                    if self.nn.opt!=None:
                        pass
                    gradient=tape.gradient(loss,self.nn.param[t])
                    self.nn.opt[t].apply_gradients(zip(gradient,self.nn.param[t]))
            except AttributeError:
                if self.thread==None:
                    gradient=self.nn.gradient(tape,loss,self.nn.param)
                    self.nn.oopt(gradient,self.nn.param)
                else:
                    gradient=self.nn.gradient(tape,loss,self.nn.param[t])
                    self.nn.oopt(gradient,self.nn.param,t)
        else:
            if self.thread_lock!=None:
                try:
                    if self.nn.opt!=None:
                        pass
                    if self.PO==1:
                        self.thread_lock[0].acquire()
                        gradient=tape.gradient(loss,self.nn.param)
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                        self.thread_lock[0].release()
                    else:
                        self.thread_lock[0].acquire()
                        self.param=self.nn.param
                        self.gradient=tape.gradient(loss,self.param)
                        self.thread_lock[0].release()
                        self.thread_lock[1].acquire()
                        self.nn.opt.apply_gradients(zip(self.gradient,self.nn.param))
                        self.thread_lock[1].release()
                except AttributeError:
                    if self.PO==1:
                        self.thread_lock[0].acquire()
                        gradient=self.nn.gradient(tape,loss,self.nn.param)
                        self.nn.oopt(gradient,self.nn.param)
                        self.thread_lock[0].release()
                    else:
                        self.thread_lock[0].acquire()
                        self.param=self.nn.param
                        self.gradient=self.nn.gradient(tape,loss,self.param)
                        self.thread_lock[0].release()
                        self.thread_lock[1].acquire()
                        self.nn.oopt(self.gradient,self.nn.param)
                        self.thread_lock[1].release()
            else:
                try:
                    if self.nn.opt!=None:
                        pass
                    gradient=tape.gradient(loss,self.nn.param)
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                    gradient=self.nn.gradient(tape,loss,self.nn.param)
                    self.nn.oopt(gradient,self.nn.param)
        return output,loss
    
    
    @function
    def tf_opt_t(self,data,labels,t):
        try:
            if self.nn.GradientTape!=None:
                pass
            tape,output,loss=self.nn.GradientTape(data,labels)
        except AttributeError:
            with self.core.GradientTape(persistent=True) as tape:
                try:
                    output=self.nn.fp(data)
                    loss=self.nn.loss(output,labels)
                except TypeError:
                    output,loss=self.nn.fp(data,labels)
        if self.PO==1:
            self.thread_lock[0].acquire()
            try:
                if self.nn.opt!=None:
                    pass
                self.gradient=tape.gradient(loss,self.nn.param)
                self.nn.opt.apply_gradients(zip(self.gradient,self.nn.param))
            except AttributeError:
                self.gradient=self.nn.gradient(tape,loss,self.nn.param)
                try:
                    self.nn.oopt(self.gradient,self.nn.param,t)
                except TypeError:
                    self.nn.oopt(self.gradient,self.nn.param)
            self.thread_lock[0].release()
        else:
            self.thread_lock[0].acquire()
            try:
                if self.nn.opt!=None:
                    pass
                self.param=self.nn.param
                self.gradient=tape.gradient(loss,self.param)
            except AttributeError:
                self.gradient=self.nn.gradient(tape,loss,self.param)
            self.thread_lock[0].release()
            self.thread_lock[1].acquire()
            try:
                if self.nn.opt!=None:
                    pass
                self.nn.opt.apply_gradients(zip(self.gradient,self.nn.param))
            except AttributeError:
                try:
                    self.nn.oopt(self.gradient,self.nn.param,t)
                except TypeError:
                    self.nn.oopt(self.gradient,self.nn.param)
            self.thread_lock[1].release()
        return output,loss
    
    
    def opt(self,data,labels,t=None):
        if t!=None:
            t=int(t)
        try:
            if self.core.DType!=None:
                pass
            output,loss=self.tf_opt(data,labels,t)
        except AttributeError:
            if self.thread==None:
                output=self.nn.fp(data.to(self.nn.device))
            else:
                output=self.nn.fp(data.to(self.nn.device),t)
            loss=self.nn.loss(output,labels.to(self.nn.device))
            try:
                self.nn.opt.zero_grad()
                loss.backward()
                self.nn.opt.step()
            except:
                self.nn.opt(loss)
        return output,loss
    
    
    def opt_t(self,data,labels,t):
        try:
            if self.core.DType!=None:
                pass
            output,loss=self.tf_opt_t(data,labels,int(t))
        except AttributeError:
            try:
                output=self.nn.fp(data.to(self.nn.device))
            except:
                output=self.nn.fp(data.to(self.nn.device),t)
            loss=self.nn.loss(output,labels.to(self.nn.device))
            if self.PO==1:
                try:
                    self.thread_lock[0].acquire()
                    self.nn.opt.zero_grad()
                    loss.backward()
                    self.nn.opt.step()
                    self.thread_lock[0].release()
                except:
                    self.thread_lock[0].acquire()
                    try:
                        self.nn.opt(loss)
                    except:
                        self.nn.opt(loss,t)
                    self.thread_lock[0].release()
        return output,loss
    
    
    def _train(self,batch=None,_data_batch=None,_labels_batch=None,test_batch=None,t=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            batches=int((self.shape0-self.shape0%batch)/batch)
            for j in range(batches):
                self.suspend_func()
                index1=j*batch
                index2=(j+1)*batch
                data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,j)
                output,batch_loss=self.opt(data_batch,labels_batch,t)
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
                self.suspend_func()
                batches+=1
                index1=batches*batch
                index2=batch-(self.shape0-batches*batch)
                data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,flag=True)
                output,batch_loss=self.opt(data_batch,labels_batch,t)
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
            try:
                if self.nn.accuracy!=None:
                    pass
                train_acc=total_acc/batches
            except AttributeError:
                pass
            if self.thread==None:
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
            else:
                self.train_loss_list[t].append(loss.astype(np.float32))
                self.train_loss[t]=loss
                self.train_loss[t]=self.train_loss[t].astype(np.float32)
            try:
                if self.nn.accuracy!=None:
                    pass
                if self.thread==None:
                    self.train_acc_list.append(train_acc.astype(np.float32))
                    self.train_acc=train_acc
                    self.train_acc=self.train_acc.astype(np.float32)
                else:
                    self.train_acc_list[t].append(train_acc.astype(np.float32))
                    self.train_acc[t]=train_acc
                    self.train_acc[t]=self.train_acc[t].astype(np.float32)
            except AttributeError:
                pass
            if self.test_flag==True:
                if self.thread==None:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            pass
                        self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                else:
                    self.test_loss[t],self.test_acc[t]=self.test(self.test_data,self.test_labels,test_batch,t)
                    self.test_loss_list[t].append(self.test_loss[t])
                    try:
                        if self.nn.accuracy!=None:
                            pass
                        self.test_acc_list[t].append(self.test_acc[t])
                    except AttributeError:
                        pass
        elif self.ol==None:
            self.suspend_func()
            output,train_loss=self.opt(self.train_data,self.train_labels,t)
            self.loss_acc(output=output,labels_batch=labels_batch,loss=train_loss,test_batch=test_batch,total_loss=total_loss,total_acc=total_acc,t=t)
        else:
            self.suspend_func()
            data=self.ol()
            output,loss=self.opt(data[0],data[1])
            loss=loss.numpy()
            if self.thread_lock!=None:
                if self.PO==1:
                    self.thread_lock[1].acquire()
                else:
                    self.thread_lock[2].acquire()
                self.nn.train_loss=loss.astype(np.float32)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                self.total_epoch+=1
                if self.PO==1:
                    self.thread_lock[1].release()
                else:
                    self.thread_lock[2].release()
            else:
                self.nn.train_loss=loss.astype(np.float32)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                self.total_epoch+=1
        return
    
    
    def train_(self,_data_batch=None,_labels_batch=None,batch=None,batches=None,test_batch=None,index1=None,index2=None,j=None,t=None):
        if batch!=None:
            if index1==batches*batch:
                data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,j,True)
                output,batch_loss=self.opt_t(data_batch,labels_batch,t)
                try:
                    self.nn.bc[t]+=1
                except AttributeError:
                    pass
                try:
                    if self.nn.accuracy!=None:
                        pass
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    return batch_loss,batch_acc
                except AttributeError:
                    return batch_loss,None
            data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,j)
            output,batch_loss=self.opt_t(data_batch,labels_batch,t)
            try:
                self.nn.bc[t]=j
            except AttributeError:
                pass
            try:
                if self.nn.accuracy!=None:
                    pass
                batch_acc=self.nn.accuracy(output,labels_batch)
                return batch_loss,batch_acc
            except AttributeError:
                return batch_loss,None
        else:
            output,train_loss=self.opt_t(self.train_data,self.train_labels,t)
            if self.PO==1:
                self.thread_lock[1].release()
                self.loss=train_loss.numpy()
                self.train_loss_list.append(self.loss.astype(np.float32))
                self.train_loss=self.loss.astype(np.float32)
                try:
                    if self.nn.accuracy!=None:
                        pass
                    self.acc=self.nn.accuracy(output,self.train_labels)
                    self.acc=self.acc.numpy()
                    self.train_acc_list.append(self.acc.astype(np.float32))
                    self.train_acc=self.acc.astype(np.float32)
                except AttributeError:
                    pass
                if self.test_flag==True:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            pass
                        self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                self.thread_lock[1].release()
            else:
                self.thread_lock[2].acquire()
                self.train_loss=train_loss.numpy()
                self.train_loss_list.append(self.train_loss.astype(np.float32))
                self.train_loss=self.train_loss.astype(np.float32)
                try:
                    if self.nn.accuracy!=None:
                        pass
                    self.acc=self.nn.accuracy(output,self.train_labels)
                    self.acc=self.acc.numpy()
                    self.train_acc_list.append(self.acc.astype(np.float32))
                    self.train_acc=self.acc.astype(np.float32)
                except AttributeError:
                    pass
                if self.test_flag==True:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            pass
                        self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                self.thread_lock[2].release()
            return
    
    
    def _train_(self,batch=None,data_batch=None,labels_batch=None,test_batch=None,t=None):
        total_loss=0
        total_acc=0
        batches=int((self.shape0-self.shape0%batch)/batch)
        if batch!=None:
            for j in range(batches):
                if self.stop==True:
                    if self.stop_func() or self.stop_flag==0:
                        return
                self.suspend_func()
                index1=j*batch
                index2=(j+1)*batch
                batch_loss,batch_acc=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
                try:
                    if self.nn.accuracy!=None:
                        pass
                    total_loss+=batch_loss
                    total_acc+=batch_acc
                except AttributeError:
                    total_loss+=batch_loss
            if self.shape0%batch!=0:
                if self.stop==True:
                    if self.stop_func() or self.stop_flag==0:
                        return
                batches+=1
                index1=batches*batch
                index2=batch-(self.shape0-batches*batch)
                batch_loss,batch_acc=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,None,t)
                try:
                    if self.nn.accuracy!=None:
                        pass
                    total_loss+=batch_loss
                    total_acc+=batch_acc
                except AttributeError:
                    total_loss+=batch_loss
            loss=total_loss.numpy()/batches
            try:
                if self.nn.accuracy!=None:
                    pass
                train_acc=total_acc.numpy()/batches
            except AttributeError:
                pass
            if self.PO==1:
                self.thread_lock[1].acquire()
            else:
                self.thread_lock[2].acquire()
            self.train_loss_list.append(loss.astype(np.float32))
            self.train_loss=loss.astype(np.float32)
            try:
                if self.nn.accuracy!=None:
                    pass
                self.train_acc_list.append(train_acc.astype(np.float32))
                self.train_acc=train_acc.astype(np.float32)
            except AttributeError:
                pass
            if self.test_flag==True:
                self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                self.test_loss_list.append(self.test_loss)
            try:
                if self.nn.accuracy!=None:
                    pass
                self.test_acc_list.append(self.test_acc)
            except AttributeError:
                pass
            if self.PO==1:
                self.thread_lock[1].release()
            else:
                self.thread_lock[2].release()
            return
        else:
            batch_loss,batch_acc=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
            return
    
    
    def train(self,batch=None,epoch=None,test_batch=None,save=None,one=True,p=None,s=None):
        if self.thread!=None:
            try:
                t=self.threadnum.pop(0)
            except IndexError:
                print('\nError,please add thread.')
                return
        if self.thread==None:
            self.training_flag=True
        elif self.thread_lock!=None:
            if self.PO==1:
                self.thread_lock[1].acquire()
            else:
                self.thread_lock[2].acquire()
            self.thread_counter+=1
            if self.PO==1:
                self.thread_lock[1].release()
            else:
                self.thread_lock[2].release()
        self.batch=batch
        self.epoch=0
        t1=None
        t2=None
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
                if self.stop==True:
                    if self.stop_func() or self.stop_flag==0:
                        return
                t1=time.time()
                if self.thread==None:
                    try:
                        self.nn.ec+=1
                    except AttributeError:
                        pass
                else:
                    try:
                        self.nn.ec[t]+=1
                    except:
                        pass
                if self.thread==None:
                    self._train(batch,data_batch,labels_batch,test_batch)
                else:
                    if self.PO!=None:
                        self._train_(batch,data_batch,labels_batch,test_batch,t)
                    else:
                        self._train(batch,data_batch,labels_batch,test_batch,t)
                if self.stop_flag==0:
                    return
                if type(self.total_epoch)!=list:
                    if self.thread_lock!=None:
                        if self.PO==1:
                            self.thread_lock[1].acquire()
                        else:
                            self.thread_lock[2].acquire()
                        self.total_epoch+=1
                        if self.PO==1:
                            self.thread_lock[1].release()
                        else:
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
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))
                                    print()
                                except AttributeError:
                                    print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                    print()
                            else:
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc))
                                    print()
                                except AttributeError:   
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                    print()
                        else:
                            if self.test_flag==False:
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f}'.format(self.total_epoch,self.train_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.6f}'.format(self.total_epoch,self.train_acc))
                                    print()
                                except AttributeError:
                                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                                    print()
                            else:
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss,self.test_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(self.total_epoch,self.train_acc*100,self.test_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(self.total_epoch,self.train_acc,self.test_acc))
                                    print()
                                except AttributeError:   
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss,self.test_loss))
                                    print()
                    if save!=None and i%s==0:
                        self.save(self.total_epoch,one)
                t2=time.time()
                if type(self.time)!=list:
                    self.time+=(t2-t1)
                else:
                    self.time[t]+=(t2-t1)
        elif self.ol==None:
            self.suspend_func()
            i=0
            while True:
                if self.stop==True:
                    if self.stop_func() or self.stop_flag==0:
                        return
                t1=time.time()
                if self.thread==None:
                    self._train(test_batch=test_batch)
                else:
                    t=self.t.pop()
                    if self.PO!=None:
                        self._train_(test_batch=test_batch,t=t)
                    else:
                        self._train(test_batch=test_batch,t=t)
                if self.stop_flag==0:
                    return
                i+=1
                if type(self.total_epoch)!=list:
                    if self.thread_lock!=None:
                        if self.PO==1:
                            self.thread_lock[1].acquire()
                        else:
                            self.thread_lock[2].acquire()
                        self.total_epoch+=1
                        if self.PO==1:
                            self.thread_lock[1].release()
                        else:
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
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))
                                    print()
                                except AttributeError:
                                    print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                    print()
                            else:
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc))
                                    print()
                                except AttributeError:   
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                    print()
                        else:
                            if self.test_flag==False:
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f}'.format(self.total_epoch,self.train_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.6f}'.format(self.total_epoch,self.train_acc))
                                    print()
                                except AttributeError:
                                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                                    print()
                            else:
                                try:
                                    if self.nn.accuracy!=None:
                                        pass
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss,self.test_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(self.total_epoch,self.train_acc*100,self.test_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(self.total_epoch,self.train_acc,self.test_acc))
                                    print()
                                except AttributeError:   
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss,self.test_loss))
                                    print()
                    if save!=None and i%s==0:
                        self.save(self.total_epoch,one)
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
                t2=time.time()
                if type(self.time)!=list:
                    self.time+=(t2-t1)
                else:
                    self.time[t]+=(t2-t1)
        else:
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
            if self.test_flag==False:
                print('last loss:{0:.6f}'.format(self.train_loss))
            else:
                print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))
            try:
                if self.nn.accuracy!=None:
                    pass
                if self.acc_flag=='%':
                    if self.test_flag==False:
                        print('last accuracy:{0:.1f}'.format(self.train_acc*100))
                    else:
                        print('last accuracy:{0:.1f},last test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))
                else:
                    if self.test_flag==False:
                        print('last accuracy:{0:.6f}'.format(self.train_acc))
                    else:
                        print('last accuracy:{0:.6f},last test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))   
            except AttributeError:
                pass
            print()
            print('time:{0}s'.format(self.total_time))
        if self.thread==None:
            self.training_flag=False
        if self.thread_lock!=None:
            if self.PO==1:
                self.thread_lock[1].acquire()
            else:
                self.thread_lock[2].acquire()
            self.thread_counter-=1
            if self.PO==1:
                self.thread_lock[1].release()
            else:
                self.thread_lock[2].release()
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
                if self.thread==None or t==None:
                    output=self.nn.fp(data_batch)
                else:
                    output=self.nn.fp(data_batch,t)
                batch_loss=self.nn.loss(output,labels_batch)
                total_loss+=batch_loss
                try:
                    if self.nn.accuracy!=None:
                        pass
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    total_acc+=batch_acc
                except AttributeError:
                    pass
            if shape0%batch!=0:
                batches+=1
                index1=batches*batch
                index2=batch-(shape0-batches*batch)
                try:
                    if type(test_data)==list:
                        for i in range(len(test_data)):
                            data_batch[i]=self.core.concat([test_data[i][index1:],test_data[i][:index2]],0)
                    else:
                        data_batch=self.core.concat([test_data[index1:],test_data[:index2]],0)
                    if type(self.test_labels)==list:
                        for i in range(len(test_labels)):
                            labels_batch[i]=self.core.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                    else:
                        labels_batch=self.core.concat([test_labels[index1:],test_labels[:index2]],0)
                except:
                    if type(test_data)==list:
                        for i in range(len(test_data)):
                            data_batch[i]=self.core.concat([test_data[i][index1:],test_data[i][:index2]],0)
                    else:
                        data_batch=self.core.concat([test_data[index1:],test_data[:index2]],0)
                    if type(self.test_labels)==list:
                        for i in range(len(test_labels)):
                            labels_batch[i]=self.core.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                    else:
                        labels_batch=self.core.concat([test_labels[index1:],test_labels[:index2]],0)
                if self.thread==None or t==None:
                    output=self.nn.fp(data_batch)
                else:
                    output=self.nn.fp(data_batch,t)
                batch_loss=self.nn.loss(output,labels_batch)
                total_loss+=batch_loss
                try:
                    if self.nn.accuracy!=None:
                        pass
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    total_acc+=batch_acc
                except AttributeError:
                    pass
            test_loss=total_loss.numpy()/batches
            test_loss=test_loss
            test_loss=test_loss.astype(np.float32)
            try:
                if self.nn.accuracy!=None:
                    pass
                test_acc=total_acc.numpy()/batches
                test_acc=test_acc
                test_acc=test_acc.astype(np.float32)
            except AttributeError:
                pass
        else:
            if self.thread==None or t==None:
                output=self.nn.fp(test_data)
            else:
                output=self.nn.fp(test_data,t)
            test_loss=self.nn.loss(output,test_labels)
            test_loss=test_loss.numpy().astype(np.float32)
            try:
                if self.nn.accuracy!=None:
                    pass
                test_acc=self.nn.accuracy(output,test_labels)
                test_acc=test_acc.numpy().astype(np.float32)
            except AttributeError:
                pass
        if self.thread==None or t==None:
            print('test loss:{0:.6f}'.format(test_loss))
            try:
                if self.nn.accuracy!=None:
                    pass
                if self.acc_flag=='%':
                    print('accuracy:{0:.1f}'.format(test_acc*100))
                else:
                    print('accuracy:{0:.6f}'.format(test_acc))
                if self.acc_flag=='%':
                    return test_loss,test_acc*100
                else:
                    return test_loss,test_acc
            except AttributeError:
                return test_loss
    
    
    def suspend_func(self):
        if self.suspend==True:
            if self.thread==None:
                if self.save_epoch==None:
                    print('Training have suspended.')
                else:
                    self._save()
            while True:
                if self.suspend==False:
                    if self.thread==None:
                        print('Training have continued.')
                    break
        return
    
    
    def stop_func(self):
        if self.thread_lock==None:
            if self.end():
                self.training_flag=False
                self.save(self.total_epoch,True)
                print('\nSystem have stopped training,Neural network have been saved.')
                self._time=self.time-int(self.time)
                if self._time<0.5:
                    self.time=int(self.time)
                else:
                    self.time=int(self.time)+1
                self.total_time+=self.time
                print()
                print('epoch:{0}'.format(self.total_epoch))
                if self.test_flag==False:
                    print('last loss:{0:.6f}'.format(self.train_loss))
                else:
                    print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))
                try:
                    if self.nn.accuracy!=None:
                        pass
                    if self.acc_flag=='%':
                        if self.test_flag==False:
                            print('last accuracy:{0:.1f}'.format(self.train_acc*100))
                        else:
                            print('last accuracy:{0:.1f},last test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))
                    else:
                        if self.test_flag==False:
                            print('last accuracy:{0:.6f}'.format(self.train_acc))
                        else:
                            print('last accuracy:{0:.6f},last test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))   
                except AttributeError:
                    pass
                print()
                print('time:{0}s'.format(self.total_time))
                self.stop_flag=0
                return True
            elif self.stop_flag==1:
                print('\nSystem have stopped training.')
                self._time=self.time-int(self.time)
                if self._time<0.5:
                    self.time=int(self.time)
                else:
                    self.time=int(self.time)+1
                self.total_time+=self.time
                print()
                print('epoch:{0}'.format(self.total_epoch))
                if self.test_flag==False:
                    print('last loss:{0:.6f}'.format(self.train_loss))
                else:
                    print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))
                try:
                    if self.nn.accuracy!=None:
                        pass
                    if self.acc_flag=='%':
                        if self.test_flag==False:
                            print('last accuracy:{0:.1f}'.format(self.train_acc*100))
                        else:
                            print('last accuracy:{0:.1f},last test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))
                    else:
                        if self.test_flag==False:
                            print('last accuracy:{0:.6f}'.format(self.train_acc))
                        else:
                            print('last accuracy:{0:.6f},last test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))   
                except AttributeError:
                    pass
                print()
                print('time:{0}s'.format(self.total_time))
                self.stop_flag=0
                return True
        else:
            if self.end():
                if self.PO==1:
                   self.thread_lock[2].acquire()
                else:
                    self.thread_lock[3].acquire()
                self.save(self.total_epoch,True)
                self.save_flag=True
                if self.PO==1:
                    self.thread_lock[2].release()
                else:
                    self.thread_lock[3].release()
                self.stop_flag=0
                return True
            elif self.stop_flag==1:
                self.stop_flag=0
                return True
        return False
    
    
    def _save(self):
        if self.save_epoch==self.total_epoch:
            self.save(self.total_epoch,False)
            self.save_epoch=None
            print('\nNeural network have saved and training have suspended.')
            return
        elif self.save_epoch!=None and self.save_epoch>self.total_epoch:
            print('\nsave_epoch>total_epoch')
        return
    
    
    def train_info(self):
        params=1
        total_params=0
        for i in range(len(self.nn.param)):
            for j in range(len(self.nn.param[i].shape)):
                params*=self.nn.param[i].shape[j]
            total_params+=params
            params=1
        print()
        print('total params:{0}'.format(total_params))
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
        if self.acc_flag=='%':
            print('train acc:{0:.1f}'.format(self.train_acc*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc))       
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        if self.acc_flag=='%':
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
        print('train loss:{0:.6f}'.format(self.train_loss))
        try:
            if self.nn.accuracy!=None:
                pass
            plt.figure(2)
            plt.plot(np.arange(self.total_epoch),self.train_acc_list)
            plt.title('train acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            if self.acc_flag=='%':
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
        print('test loss:{0:.6f}'.format(self.test_loss))
        try:
            if self.nn.accuracy!=None:
                pass
            plt.figure(2)
            plt.plot(np.arange(self.total_epoch),self.test_acc_list)
            plt.title('test acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            if self.acc_flag=='%':
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
        print('train loss:{0}'.format(self.train_loss))
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
            if self.acc_flag=='%':
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
            if self.acc_flag=='%':
                print('test acc:{0:.1f}'.format(self.test_acc*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc)) 
        return
    
    
    def save_p(self):
        parameter_file=open('param.dat','wb')
        pickle.dump(self.nn.param,parameter_file)
        parameter_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if self.save_flag==True:
            return
        if one==True:
            output_file=open('save.dat','wb')
            try:
                if len(self.nn.model.weights)==self.nn.param:
                    pass
                else:
                    parameter_file=open('param.dat','wb')
            except AttributeError:
                try:
                    if self.core.DType!=None:
                        pass 
                    parameter_file=open('param.dat','wb')
                except AttributeError:
                    pass
        else:
            output_file=open('save-{0}.dat'.format(i),'wb')
            try:
                if len(self.nn.model.weights)==self.nn.param:
                    self.file_list.append(['save-{0}.dat'])
                    if len(self.file_list)>self.s+1:
                        os.remove(self.file_list[0][0])
                        del self.file_list[0]
                else:
                    parameter_file=open('param-{0}.dat'.format(i),'wb')
                    self.file_list.append(['save-{0}.dat','param-{0}.dat'])
                    if len(self.file_list)>self.s+1:
                        os.remove(self.file_list[0][0])
                        os.remove(self.file_list[0][1])
                        del self.file_list[0]
            except AttributeError:
                try:
                    if self.core.DType!=None:
                        pass   
                    parameter_file=open('param-{0}.dat'.format(i),'wb')
                    self.file_list.append(['save-{0}.dat','param-{0}.dat'])
                    if len(self.file_list)>self.s+1:
                        os.remove(self.file_list[0][0])
                        os.remove(self.file_list[0][1])
                        del self.file_list[0]
                except AttributeError:
                    self.file_list.append(['save-{0}.dat'])
                    if len(self.file_list)>self.s+1:
                        os.remove(self.file_list[0][0])
                        del self.file_list[0]
        try:
            if self.core.DType!=None:
                pass  
            try:
                if len(self.nn.model.weights)!=self.nn.param:
                    pickle.dump(self.nn.param[:-len(self.nn.model.weights)],parameter_file)
            except AttributeError:
                pickle.dump(self.nn.param,parameter_file)
            if self.training_flag==False:
                self.nn.param=None
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
            pickle.dump(opt.get_config(),output_file)
        except AttributeError:
            pass
        pickle.dump(self.ol,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.end_acc,output_file)
        pickle.dump(self.end_test_loss,output_file)
        pickle.dump(self.end_test_acc,output_file)
        pickle.dump(self.acc_flag,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.s,output_file)
        pickle.dump(self.file_list,output_file)
        pickle.dump(self.train_counter,output_file)
        pickle.dump(self.thread_counter,output_file)
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
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        try:
            if len(self.nn.model.weights)==self.nn.param:
                pass
            else:
                parameter_file.close()
        except AttributeError:
            try:
                if self.core.DType!=None:
                    pass  
                parameter_file.close()
            except AttributeError:
                pass
        if self.save_flag==True:
            print('\nSystem have stopped,Neural network have saved.')
        return
    
	
    def restore(self,s_path,p_path=None):
        input_file=open(s_path,'rb')
        if p_path!=None:
            parameter_file=open(p_path,'rb')
            param=pickle.load(parameter_file)
        self.nn=pickle.load(input_file)
        try:
            if self.core.DType!=None:
                pass 
            try:
                if self.nn.model!=None:
                    pass
                self.nn.param=param.extend(self.nn.model.weights)
            except AttributeError:
                self.nn.param=param
            try:
                self.nn.km=1
            except AttributeError:
                pass
            self.config=pickle.load(input_file)
        except AttributeError:
            pass
        self.ol=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.end_acc=pickle.load(input_file)
        self.end_test_loss=pickle.load(input_file)
        self.end_test_acc=pickle.load(input_file)
        self.acc_flag=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.s=pickle.load(input_file)
        self.file_list=pickle.load(input_file)
        self.train_counter=pickle.load(input_file)
        self.thread_counter=pickle.load(input_file)
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
        if p_path!=None:
            parameter_file.close()
        return
