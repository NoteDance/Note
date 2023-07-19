import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Process,Value,Array
import numpy as np
from Note.DL.dl.test import parallel_test
import matplotlib.pyplot as plt
import pickle
import os


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        try:
            self.nn.km=1
        except Exception:
            pass
        self.PO=None
        self.process=None
        self.process_t=None
        self.train_ds=None
        self.prefetch_batch_size=tf.data.AUTOTUNE
        self.prefetch_batch_size_t=tf.data.AUTOTUNE
        self.data_segment_flag=False
        self.batches=None
        self.buffer_size=None
        self.priority_flag=False
        self.priority_p=0
        self.max_opt=None
        self.epoch=None
        self.epoch_counter=0
        self.stop=False
        self.stop_flag=False
        self.save_flag=False
        self.save_epoch=None
        self.batch=None
        self.epoch_=0
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.acc_flag='%'
        self.opt_counter=None
        self.p=None
        self.s=None
        self.saving_one=True
        self.filename='save.dat'
        self.train_loss=0
        self.train_acc=0
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=0
        self.test_acc=0
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.test_flag=False
        self.total_epoch=0
    
    
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        if type(self.nn.param[0])!=list:
            self.train_data=train_data.astype(self.nn.param[0].dtype.name)
            self.train_labels=train_labels.astype(self.nn.param[0].dtype.name)
        else:
            self.train_data=train_data.astype(self.nn.param[0][0].dtype.name)
            self.train_labels=train_labels.astype(self.nn.param[0][0].dtype.name)
        self.train_dataset=train_dataset
        if test_data is not None:
            self.test_data=test_data
            self.test_labels=test_labels
            self.test_flag=True
        self.test_dataset=test_dataset
        self.batch_counter=np.zeros(self.process,dtype=np.int32)
        if type(self.nn.param[0])!=list:
            self.total_loss=np.zeros(self.process,dtype=self.nn.param[0].dtype.name)
        else:
            self.total_loss=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name)
        try:
            if self.nn.accuracy!=None:
                if type(self.nn.param[0])!=list:
                    self.total_acc=np.zeros(self.process,dtype=self.nn.param[0].dtype.name)
                else:
                    self.total_acc=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name)
        except Exception:
            pass
        if self.priority_flag==True:
            self.opt_counter=np.zeros(self.process,dtype=np.int32)
        if self.train_dataset==None:
            if type(self.train_data)==list:
                self.shape0=train_data[0].shape[0]
                self.batches=int((self.shape0-self.shape0%self.batch)/self.batch)
                if self.shape0%self.batch!=0:
                    self.batches+=1
            else:
                self.shape0=train_data.shape[0]
                self.batches=int((self.shape0-self.shape0%self.batch)/self.batch)
                if self.shape0%self.batch!=0:
                    self.batches+=1
        if self.data_segment_flag==True:
            self.train_data,self.train_labels=self.segment_data()
        return
    
    
    def segment_data(self):
        if len(self.train_data)!=self.process:
            segments=int((len(self.train_data)-len(self.train_data)%self.process)/self.process)
            for i in range(self.process):
                index1=i*segments
                index2=(i+1)*segments
                if i==0:
                    data=np.expand_dims(self.train_data[index1:index2],axis=0)
                    labels=np.expand_dims(self.train_labels[index1:index2],axis=0)
                else:
                    data=np.concatenate((data,np.expand_dims(self.train_data[index1:index2],axis=0)))
                    labels=np.concatenate((labels,np.expand_dims(self.train_labels[index1:index2],axis=0)))
            if len(data)%self.process!=0:
                segments+=1
                index1=segments*self.process
                index2=self.process-(len(self.train_data)-segments*self.process)
                data=np.concatenate((data,np.expand_dims(self.train_data[index1:index2],axis=0)))
                labels=np.concatenate((labels,np.expand_dims(self.train_labels[index1:index2],axis=0)))
            return data,labels
                
    
    def init(self,manager):
        self.epoch_counter=Value('i',self.epoch_counter)
        self.batch_counter=Array('i',self.batch_counter)
        self.total_loss=Array('f',self.total_loss)
        self.total_epoch=Value('i',self.total_epoch)
        self.train_loss=Value('f',self.train_loss)
        self.train_loss_list=manager.list(self.train_loss_list)
        self.priority_p=Value('i',self.priority_p)
        if self.test_flag==True:
            self.test_loss=Value('f',self.test_loss)
            self.test_loss_list=manager.list(self.test_loss_list)
        try:
            if self.nn.accuracy!=None:
                self.total_acc=Array('f',self.total_acc)
                self.train_acc=Value('f',self.train_acc)
                self.train_acc_list=manager.list(self.train_acc_list)
                if self.test_flag==True:
                    self.test_acc=Value('f',self.test_acc)
                    self.test_acc_list=manager.list(self.test_acc_list)
        except Exception:
            pass
        if self.priority_flag==True:
            self.opt_counter=Array('i',self.opt_counter)  
        try:
            if self.nn.attenuate!=None:
              self.nn.opt_counter=manager.list([self.nn.opt_counter])  
        except Exception:
            pass
        try:
            self.nn.ec=manager.list([self.nn.ec])  
        except Exception:
            pass
        try:
            self.nn.bc=manager.list([self.nn.bc])
        except Exception:
            pass
        self.epoch_=Value('i',self.epoch_)
        self.stop_flag=Value('b',self.stop_flag)
        self.save_flag=Value('b',self.save_flag)
        self.file_list=manager.list([])
        self.param=manager.dict()
        self.param[7]=self.nn.param
        return
    
    
    def init_online(self,manager):
        self.nn.train_loss_list=manager.list([])
        self.nn.train_acc_list=manager.list([])
        self.nn.counter=manager.list([])
        self.nn.exception_list=manager.list([])
        self.param=manager.dict()
        self.param[7]=self.nn.param
        return
    
    
    def end(self):
        if self.end_loss!=None and len(self.train_loss_list)!=0 and self.train_loss_list[-1]<self.end_loss:
            return True
        elif self.end_acc!=None and len(self.train_acc_list)!=0 and self.train_acc_list[-1]>self.end_acc:
            return True
        elif self.end_loss!=None and len(self.train_loss_list)!=0 and self.end_acc!=None and self.train_loss_list[-1]<self.end_loss and self.train_acc_list[-1]>self.end_acc:
            return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.test_loss_list[-1]<self.end_test_loss:
            return True
        elif self.end_test_acc!=None and len(self.test_acc_list)!=0 and self.test_acc_list[-1]>self.end_test_acc:
            return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.end_test_acc!=None and self.test_loss_list[-1]<self.end_test_loss and self.test_acc_list[-1]>self.end_test_acc:
            return True
    
    
    @tf.function(jit_compile=True)
    def opt_p(self,data,labels,p,lock,g_lock=None):
        try:
            try:
                with tf.GradientTape(persistent=True) as tape:
                    try:
                        try:
                            output=self.nn.fp(data,p)
                            loss=self.nn.loss(output,labels,p)
                        except Exception:
                            output,loss=self.nn.fp(data,labels,p)
                    except Exception:
                        try:
                            output=self.nn.fp(data)
                            loss=self.nn.loss(output,labels)
                        except Exception:
                            output,loss=self.nn.fp(data,labels)
            except Exception:
                if self.nn.GradientTape!=None:
                    tape,output,loss=self.nn.GradientTape(data,labels,p)
        except Exception as e:
            raise e
        if self.PO==1:
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return None,0
            try:
                try:
                    try:
                        gradient=self.nn.gradient(tape,loss)
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])
                except Exception:
                    gradient=tape.gradient(loss,self.nn.param)
            except Exception as e:
                raise e
            try:
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)
            except Exception as e:
                try:
                    if self.nn.attenuate!=None:
                        raise e
                except Exception:
                    pass
            try:
                try:
                    param=self.nn.opt(gradient)
                except Exception:
                    param=self.nn.opt(gradient,p)
            except Exception as e:
                raise e
            lock[0].release()
        elif self.PO==2:
            g_lock.acquire()
            if self.stop_func_(g_lock):
                return None,0
            try:
                try:
                    try:
                        gradient=self.nn.gradient(tape,loss)
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])
                except Exception:
                    gradient=tape.gradient(loss,self.nn.param)
            except Exception as e:
                raise e
            g_lock.release()
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return None,0
            try:
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)
            except Exception as e:
                try:
                    if self.nn.attenuate!=None:
                        raise e
                except Exception:
                    pass
            try:
                try:
                    param=self.nn.opt(gradient)
                except Exception:
                    param=self.nn.opt(gradient,p)
            except Exception as e:
                raise e
            lock[0].release()
        elif self.PO==3:
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            if self.stop_func_():
                return None,0
            try:
                try:
                    try:
                        gradient=self.nn.gradient(tape,loss)
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])
                except Exception:
                    gradient=tape.gradient(loss,self.nn.param)
            except Exception as e:
                raise e
            try:
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)
            except Exception as e:
                try:
                    if self.nn.attenuate!=None:
                        raise e
                except Exception:
                    pass
            try:
                try:
                    param=self.nn.opt(gradient)
                except Exception:
                    param=self.nn.opt(gradient,p)
            except Exception as e:
                raise e
        return output,loss,param
    
    
    def opt(self,data,labels,p,lock,g_lock):
        if self.PO==2:
            if type(g_lock)!=list:
                pass
            elif len(g_lock)==self.process:
                ln=p
                g_lock=g_lock[ln]
            else:
                ln=int(np.random.choice(len(g_lock)))
                g_lock=g_lock[ln]
            output,loss,param=self.opt_p(data,labels,p,lock,g_lock)
        else:
            output,loss,param=self.opt_p(data,labels,p,lock)
        return output,loss,param
    
    
    def update_nn_param(self,param=None):
        if param==None:
            parameter_flat=nest.flatten(self.nn.param)
            parameter7_flat=nest.flatten(self.param[7])
        else:
            parameter_flat=nest.flatten(self.nn.param)
            parameter7_flat=nest.flatten(param)
        for i in range(len(parameter_flat)):
            if param==None:
                state_ops.assign(parameter_flat[i],parameter7_flat[i])
            else:
                state_ops.assign(parameter_flat[i],parameter7_flat[i])
        self.nn.param=nest.pack_sequence_as(self.nn.param,parameter_flat)
        self.param[7]=nest.pack_sequence_as(self.param[7],parameter7_flat)
        return
    
    
    def train7(self,train_ds,p,test_batch,lock,g_lock):
        while True:
            for data_batch,labels_batch in train_ds:
                try:
                    data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)
                except Exception as e:
                    try:
                        if self.nn.data_func!=None:
                            raise e
                    except Exception:
                        pass
                if self.priority_flag==True:
                    self.priority_p.value=np.argmax(self.opt_counter)
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt:
                        self.priority_p.value=int(self.priority_p.value)
                    elif self.max_opt==None:
                        self.priority_p.value=int(self.priority_p.value)
                    else:
                        self.priority_p.value=-1
                if self.priority_flag==True:
                    self.opt_counter[p]=0
                try:
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter.scatter_update(tf.IndexedSlices(0,p))
                    self.nn.opt_counter[0]=opt_counter
                except Exception as e:
                    try:
                       if self.nn.attenuate!=None:
                           raise e
                    except Exception:
                        pass
                output,batch_loss,param=self.opt(data_batch,labels_batch,p,lock,g_lock)
                self.param[7]=param
                if self.priority_flag==True:
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')
                    opt_counter+=1
                try:
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter.assign(opt_counter+1)
                    self.nn.opt_counter[0]=opt_counter
                except Exception as e:
                    try:
                       if self.nn.attenuate!=None:
                           raise e
                    except Exception:
                        pass
                try:
                    bc=self.nn.bc[0]
                    bc.assign_add(1)
                    self.nn.bc[0]=bc
                except Exception:
                    pass
                try:
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch,p)
                    except Exception:
                        batch_acc=self.nn.accuracy(output,labels_batch)
                except Exception as e:
                    try:
                        if self.nn.accuracy!=None:
                           raise e
                    except Exception:
                       pass
                try:
                    if self.nn.accuracy!=None:
                        self.total_loss[p]+=batch_loss
                        self.total_acc[p]+=batch_acc
                except Exception:
                    self.total_loss[p]+=batch_loss
                self.batch_counter[p]+=1
                if self.PO==1 or self.PO==2:
                    lock[1].acquire()
                elif lock!=None:
                    lock.acquire()
                batches=np.sum(self.batch_counter)
                if batches>=self.batches:
                    batch_counter=np.frombuffer(self.batch_counter.get_obj(),dtype='i')
                    batch_counter*=0
                    loss=np.sum(self.total_loss)/batches
                    try:
                        if self.nn.accuracy!=None:
                            train_acc=np.sum(self.total_acc)/batches
                    except Exception:
                        pass
                    self.total_epoch.value+=1
                    self.train_loss.value=loss
                    self.train_loss_list.append(loss)
                    try:
                        if self.nn.accuracy!=None:
                            self.train_acc.value=train_acc
                            self.train_acc_list.append(train_acc)
                    except Exception:
                        pass
                    if self.test_flag==True:
                        try:
                            try:
                                if self.nn.accuracy!=None:
                                    self.test_loss.value,self.test_acc.value=self.test(self.test_data,self.test_labels,test_batch,p)
                                    self.test_loss_list.append(self.test_loss.value)
                                    self.test_acc_list.append(self.test_acc.value)
                            except Exception:
                                self.test_loss.value=self.test(self.test_data,self.test_labels,test_batch,p)
                                self.test_loss_list.append(self.test_loss.value)
                        except Exception as e:
                            raise e
                    self.print_save()
                    self.epoch_counter.value+=1
                    try:
                        ec=self.nn.ec[0]
                        ec.assign_add(1)
                        self.nn.ec[0]=ec
                    except Exception:
                        pass
                    total_loss=np.frombuffer(self.total_loss.get_obj(),dtype='f')
                    total_loss*=0
                    try:
                        total_acc=np.frombuffer(self.total_acc.get_obj(),dtype='f')
                        total_acc*=0
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:
                                raise e
                        except Exception:
                            pass
                if self.PO==1 or self.PO==2:
                    lock[1].release()
                elif lock!=None:
                    lock.release()
                if self.epoch_counter.value>=self.epoch:
                    self.param[7]=param
                    return
    
    
    def train(self,p,lock=None,g_lock=None,test_batch=None):
        if self.epoch!=None:
            if self.train_dataset!=None:
                train_ds=self.train_dataset
            else:
                if self.data_segment_flag==True:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data[p],self.train_labels[p])).batch(self.batch).prefetch(self.prefetch_batch_size)
                elif self.buffer_size!=None:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).shuffle(self.buffer_size).batch(self.batch).prefetch(self.prefetch_batch_size)
                else:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).batch(self.batch).prefetch(self.prefetch_batch_size)
        self.train7(train_ds,p,test_batch,lock,g_lock)
        return
    
    
    def train_online(self,p,lock=None,g_lock=None):
        self.nn.counter.append(0)
        while True:
            try:
                self.nn.save(self.save,p)
            except AttributeError:
                pass
            try:
                if self.nn.stop_flag==True:
                    return
            except AttributeError:
                pass
            try:
                if self.nn.stop_func(p):
                    return
            except AttributeError:
                pass
            try:
                self.nn.suspend_func(p)
            except AttributeError:
                pass
            try:
                data=self.nn.online(p)
            except Exception as e:
                self.nn.exception_list[p]=e
            if data=='stop':
                return
            elif data=='suspend':
                self.nn.suspend_func(p)
            try:
                if self.PO==2:
                    if type(g_lock)!=list:
                        pass
                    elif len(g_lock)==self.process:
                        ln=p
                        g_lock=g_lock[ln]
                    else:
                        ln=int(np.random.choice(len(g_lock)))
                        g_lock=g_lock[ln]
                output,loss,param=self.opt(data[0],data[1],p,lock,g_lock)
                self.param[7]=param
            except Exception as e:
                if self.PO==1:
                    if lock[0].acquire(False):
                        lock[0].release()
                elif self.PO==2:
                    if g_lock.acquire(False):
                        g_lock.release()
                    if lock[0].acquire(False):
                        lock[0].release()
                self.nn.exception_list[p]=e
            loss=loss.numpy()
            if len(self.nn.train_loss_list)==self.nn.max_length:
                del self.nn.train_loss_list[0]
            self.nn.train_loss_list.append(loss)
            try:
                try:
                    acc=self.nn.accuracy(output,data[1])
                except Exception:
                    self.exception_list[p]=True
                if len(self.nn.train_acc_list)==self.nn.max_length:
                    del self.nn.train_acc_list[0]
                self.nn.train_acc_list.append(acc)
            except Exception as e:
                try:
                    if self.nn.accuracy!=None:
                        self.nn.exception_list[p]=e
                except Exception:
                    pass
            try:
                count=self.nn.counter[p]
                count+=1
                self.nn.counter[p]=count
            except IndexError:
                self.nn.counter.append(0)
                count=self.nn.counter[p]
                count+=1
                self.nn.counter[p]=count
        return
    
    
    @tf.function(jit_compile=True)
    def test_(self,data,labels):
        try:
            try:
                output=self.nn.fp(data)
                loss=self.nn.loss(output,labels)
            except Exception:
                output,loss=self.nn.fp(data,labels)
        except Exception as e:
            raise e
        try:
            acc=self.nn.accuracy(output,labels)
        except Exception as e:
            try:
                if self.nn.accuracy!=None:
                    raise e
            except Exception:
                acc=None
                pass
        return loss,acc
    
    
    def test(self,test_data=None,test_labels=None,batch=None,p=None):
        if type(self.nn.param[0])!=list:
            test_data=test_data.astype(self.nn.param[0].dtype.name)
            test_labels=test_labels.astype(self.nn.param[0].dtype.name)
        else:
            test_data=test_data.astype(self.nn.param[0][0].dtype.name)
            test_labels=test_labels.astype(self.nn.param[0][0].dtype.name)
        if self.process_t!=None:
            parallel_test_=parallel_test(self.nn,self.test_data,self.test_labels,self.process_t,batch,self.prefetch_batch_size_t)
            if type(self.test_data)!=list:
                parallel_test_.segment_data()
            for p in range(self.process_t):
            	Process(target=parallel_test_.test).start()
            try:
                test_loss,test_acc=parallel_test_.loss_acc()
            except Exception as e:
                try:
                    if self.nn.accuracy!=None:
                        raise e
                except Exception:
                    test_loss=parallel_test_.loss_acc()
        elif batch!=None:
            total_loss=0
            total_acc=0
            if self.test_dataset!=None:
                for data_batch,labels_batch in self.test_dataset:
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch)
                    total_loss+=batch_loss
                    try:
                        total_acc+=batch_acc
                    except Exception:
                        pass
            else:
                total_loss=0
                total_acc=0
                batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
                shape0=test_data.shape[0]
                for j in range(batches):
                    index1=j*batch
                    index2=(j+1)*batch
                    data_batch=test_data[index1:index2]
                    labels_batch=test_labels[index1:index2]
                    try:
                        try:
                            output=self.nn.fp(data_batch)
                        except Exception:
                            output=self.nn.fp(data_batch,p)
                    except Exception as e:
                        raise e
                    batch_loss=self.nn.loss(output,labels_batch)
                    total_loss+=batch_loss
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch)
                        total_acc+=batch_acc
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:
                                raise e
                        except Exception:
                            pass
                if shape0%batch!=0:
                    batches+=1
                    index1=batches*batch
                    index2=batch-(shape0-batches*batch)
                    data_batch=tf.concat([test_data[index1:],test_data[:index2]],0)
                    labels_batch=tf.concat([test_labels[index1:],test_labels[:index2]],0)
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch)
                    total_loss+=batch_loss
                    try:
                        total_acc+=batch_acc
                    except Exception:
                        pass
            test_loss=total_loss.numpy()/batches
            try:
                if self.nn.accuracy!=None:
                    test_acc=total_acc.numpy()/batches
            except Exception:
                pass
        else:
            batch_loss,batch_acc=self.test_(test_data,test_labels)
            test_loss=test_loss.numpy()
            try:
                test_acc=test_acc.numpy()
            except Exception:
                pass
        try:
            if self.nn.accuracy!=None:
                return test_loss,test_acc
        except Exception:
            return test_loss
    
    
    def stop_func(self):
        if self.end():
            self.save(self.total_epoch.value,True)
            self.save_flag.value=True
            self.stop_flag.value=True
            return True
        return False
    
    
    def stop_func_(self,lock=None):
        if self.stop==True:
            if self.stop_flag.value==True or self.stop_func():
                if self.PO!=3:
                    lock.release()
                return True
    
    
    def print_save(self):
        if self.epoch!=None:
            if self.p!=None:
                p_=self.p-1
                if self.epoch%10!=0:
                    p=self.epoch-self.epoch%p_
                    p=int(p/p_)
                    if p==0:
                        p=1
                else:
                    p=self.epoch/(p_+1)
                    p=int(p)
                    if p==0:
                        p=1
                if self.epoch_.value%p==0:
                    if self.test_flag==False:
                        try:
                            if self.nn.accuracy!=None:
                                print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch.value,self.train_loss.value))
                                if self.acc_flag=='%':
                                    print('epoch:{0}   accuracy:{1:.1f}'.format(self.total_epoch.value,self.train_acc.value*100))
                                else:
                                    print('epoch:{0}   accuracy:{1:.6f}'.format(self.total_epoch.value,self.train_acc.value))
                                print()
                        except Exception:
                            print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch.value,self.train_loss.value))
                            print()
                    else:
                        try:
                            if self.nn.accuracy!=None:
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch.value,self.train_loss.value,self.test_loss.value))
                                if self.acc_flag=='%':
                                    print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(self.total_epoch.value,self.train_acc.value*100,self.test_acc.value*100))
                                else:
                                    print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(self.total_epoch.value,self.train_acc.value,self.test_acc.value))
                                print()
                        except Exception:
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss.value,self.test_loss.value))
                            print()
            if self.s!=None:
                if self.s==1:
                    s_=1
                else:
                    s_=self.s-1
                if self.epoch%10!=0:
                    s=self.epoch-self.epoch%s_
                    s=int(s/s_)
                    if s==0:
                        s=1
                else:
                    s=self.epoch/(s_+1)
                    s=int(s)
                    if s==0:
                        s=1
                if self.epoch_.value%s==0:
                    if self.saving_one==True:
                        self.save(self.total_epoch.value)
                    else:
                        self.save(self.total_epoch.value,False)
            self.epoch_.value+=1
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
        print('epoch:{0}'.format(self.total_epoch.value))
        print()
        try:
            print('learning rate:{0}'.format(self.nn.lr))
            print()
        except Exception:
            pass
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0:.6f}'.format(self.train_loss.value))
        if self.acc_flag=='%':
            print('train acc:{0:.1f}'.format(self.train_acc.value*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc.value))       
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss.value))
        if self.acc_flag=='%':
            print('test acc:{0:.1f}'.format(self.test_acc.value*100))
        else:
            print('test acc:{0:.6f}'.format(self.test_acc.value))      
        return
		
    
    def info(self):
        self.train_info()
        if self.test_flag==True:
            print()
            print('-------------------------------------')
            self.test_info()
        return


    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch.value),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('train loss:{0:.6f}'.format(self.train_loss.value))
        try:
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(self.total_epoch.value),self.train_acc_list)
                plt.title('train acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                if self.acc_flag=='%':
                    print('train acc:{0:.1f}'.format(self.train_acc.value*100))
                else:
                    print('train acc:{0:.6f}'.format(self.train_acc.value)) 
        except Exception:
            pass
        return
    
    
    def visualize_test(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch.value),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('test loss:{0:.6f}'.format(self.test_loss.value))
        try:
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(self.total_epoch.value),self.test_acc_list)
                plt.title('test acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                if self.acc_flag=='%':
                    print('test acc:{0:.1f}'.format(self.test_acc.value*100))
                else:
                    print('test acc:{0:.6f}'.format(self.test_acc.value))  
        except Exception:
            pass
        return 
    
    
    def comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch.value),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch.value),self.test_loss_list,'r-',label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('train loss:{0}'.format(self.train_loss.value))
        plt.legend()
        try:
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(self.total_epoch.value),self.train_acc_list,'b-',label='train acc')
                if self.test_flag==True:
                    plt.plot(np.arange(self.total_epoch.value),self.test_acc_list,'r-',label='test acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.legend()
                if self.acc_flag=='%':
                    print('train acc:{0:.1f}'.format(self.train_acc.value*100))
                else:
                    print('train acc:{0:.6f}'.format(self.train_acc.value))
        except Exception:
            pass
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss.value))
            if self.acc_flag=='%':
                print('test acc:{0:.1f}'.format(self.test_acc.value*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc.value)) 
        return
    
    
    def save_p(self):
        parameter_file=open('param.dat','wb')
        pickle.dump(self.param[7],parameter_file)
        parameter_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if self.save_flag.value==True:
            return
        if one==True:
            output_file=open(self.filename,'wb')
        else:
            filename=self.filename.replace(self.filename[self.filename.find('.'):],'-{0}.dat'.format(i))
            output_file=open(filename,'wb')
            self.file_list.append([filename])
            if len(self.file_list)>self.s:
                os.remove(self.file_list[0][0])
                del self.file_list[0]
        self.update_nn_param()
        pickle.dump(self.nn,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.end_acc,output_file)
        pickle.dump(self.end_test_loss,output_file)
        pickle.dump(self.end_test_acc,output_file)
        pickle.dump(self.acc_flag,output_file)
        pickle.dump(self.file_list,output_file)
        pickle.dump(self.train_loss.value,output_file)
        pickle.dump(self.train_acc.value,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_acc_list,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss.value,output_file)
            pickle.dump(self.test_acc.value,output_file)
            pickle.dump(self.test_loss_list,output_file)
            pickle.dump(self.test_acc_list,output_file)
        pickle.dump(self.total_epoch.value,output_file)
        try:
            pickle.dump(self.nn.ec,output_file)
        except Exception:
            pass
        try:
            pickle.dump(self.nn.bc,output_file)
        except Exception:
            pass
        output_file.close()
        return
    
	
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        try:
            self.nn.km=1
        except Exception:
            pass
        self.param[7]=self.nn.param
        self.batch=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.end_acc=pickle.load(input_file)
        self.end_test_loss=pickle.load(input_file)
        self.end_test_acc=pickle.load(input_file)
        self.acc_flag=pickle.load(input_file)
        self.file_list=pickle.load(input_file)
        self.train_loss.value=pickle.load(input_file)
        self.train_acc.value=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_acc_list=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss.value=pickle.load(input_file)
            self.test_acc.value=pickle.load(input_file)
            self.test_loss_list=pickle.load(input_file)
            self.test_acc_list=pickle.load(input_file)
        self.total_epoch.value=pickle.load(input_file)
        self.nn.ec=pickle.load(input_file)
        self.nn.bc=pickle.load(input_file)
        input_file.close()
        return
