import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Process,Value,Array
import numpy as np
from Note.DL.dl.test import parallel_test
from Note.nn.Module import Module
import matplotlib.pyplot as plt
import pickle
import os


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        self.nn.km=1
        self.PO=3
        self.process=None
        self.process_t=None
        self.train_ds=None
        self.prefetch_batch_size=tf.data.AUTOTUNE
        self.prefetch_batch_size_t=tf.data.AUTOTUNE
        self.data_segment_flag=True
        self.batches=None
        self.buffer_size=None
        self.priority_flag=False
        self.max_opt=None
        self.epoch=None
        self.stop=False
        self.batch=None
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.acc_flag='%'
        self.path=None
        self.save_freq=1
        self.save_freq_=None
        self.max_save_files=None
        self.steps_per_execution=None
        self.monitor='val_loss'
        self.val_loss=0
        self.val_accuracy=1
        self.save_best_only=False
        self.save_param_only=False
        self.test_flag=False
    
    
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        self.train_data=train_data
        self.train_labels=train_labels
        self.train_dataset=train_dataset
        self.test_data=test_data
        self.test_labels=test_labels
        self.test_dataset=test_dataset
        if test_data is not None or test_dataset is not None:
            self.test_flag=True
        self.batch_counter=np.zeros(self.process,dtype='int32')
        self.batch_counter_=np.zeros(self.process,dtype='int32')
        self.total_loss=np.zeros(self.process,dtype='float32')
        if hasattr(self.nn,'accuracy'):
            self.total_acc=np.zeros(self.process,dtype='float32')
        if self.priority_flag==True:
            self.opt_counter=np.zeros(self.process,dtype='int32')
        if train_data is not None:
            self.shape0=train_data.shape[0]
            self.batches=int((self.shape0-self.shape0%self.batch)/self.batch)
            if self.shape0%self.batch!=0:
                self.batches+=1
        if self.data_segment_flag==True:
            self.train_data,self.train_labels=self.segment_data()
        return
    
    
    def segment_data(self):
        data=np.array_split(self.train_data,self.process)
        labels=np.array_split(self.train_labels,self.process)
        return data,labels
    
    
    def init(self,manager):
        self.epoch_counter=Value('i',0)
        self.batch_counter=Array('i',self.batch_counter)
        self.batch_counter_=Array('i',self.batch_counter_)
        self.total_loss=Array('f',self.total_loss)
        self.total_epoch=Value('i',0)
        self.train_loss=Value('f',0)
        self.train_loss_list=manager.list([])
        self.priority_p=Value('i',0)
        if self.test_flag==True:
            self.test_loss=Value('f',0)
            self.test_loss_list=manager.list([])
        if hasattr(self.nn,'accuracy'):
            self.total_acc=Array('f',self.total_acc)
            self.train_acc=Value('f',0)
            self.train_acc_list=manager.list([])
            if self.test_flag==True:
                self.test_acc=Value('f',0)
                self.test_acc_list=manager.list([])
        if self.priority_flag==True:
            self.opt_counter=Array('i',self.opt_counter)
        if self.nn is not None:
            self.nn.opt_counter=manager.list([tf.Variable(tf.zeros([self.process]))])  
        self._epoch_counter=manager.list([tf.Variable(0) for _ in range(self.process)])
        self.nn.ec=manager.list([0])
        self.ec=self.nn.ec[0]
        self._batch_counter=manager.list([tf.Variable(0) for _ in range(self.process)])
        self.nn.bc=manager.list([0])
        self.bc=self.nn.bc[0]
        self.epoch_=Value('i',0)
        self.stop_flag=Value('b',False)
        self.save_flag=Value('b',False)
        self.path_list=manager.list([])
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
        if self.end_acc!=None and len(self.train_acc_list)!=0 and self.train_acc_list[-1]>self.end_acc:
            return True
        elif self.end_loss!=None and len(self.train_loss_list)!=0 and self.train_loss_list[-1]<self.end_loss:
            return True
        elif self.end_test_acc!=None and len(self.test_acc_list)!=0 and self.test_acc_list[-1]>self.end_test_acc:
            return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.test_loss_list[-1]<self.end_test_loss:
            return True
        elif self.end_acc!=None and self.end_test_acc!=None:
            if len(self.train_acc_list)!=0 and len(self.test_acc_list)!=0 and self.train_acc_list[-1]>self.end_acc and self.test_acc_list[-1]>self.end_test_acc:
                return True
        elif self.end_loss!=None and self.end_test_loss!=None:
            if len(self.train_loss_list)!=0 and len(self.test_loss_list)!=0 and self.train_loss_list[-1]<self.end_loss and self.test_loss_list[-1]<self.end_test_loss:
                return True
    
    
    @tf.function(jit_compile=True)
    def opt_p(self,data,labels,p,lock,g_lock=None):
        try:
            if hasattr(self.nn,'GradientTape'):
                tape,output,loss=self.nn.GradientTape(data,labels,p)
            else:
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
        except Exception as e:
            raise e
        if self.PO==1:
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if self.stop_flag.value==True:
                        return None,None,None
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.steps_per_execution==None and self.stop_func_(lock[0]):
                return None,None,None
            try:
                if hasattr(self.nn,'gradient'):
                    try:
                        gradient=self.nn.gradient(tape,loss)
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])
                else:
                    gradient=tape.gradient(loss,self.nn.param)
            except Exception as e:
                raise e
            if hasattr(self.nn,'attenuate'):
                gradient=self.nn.attenuate(gradient,p)
            try:
                try:
                    param=self.nn.opt(gradient,p)
                except Exception:
                    param=self.nn.opt(gradient)
            except Exception as e:
                raise e
            lock[0].release()
        elif self.PO==2:
            g_lock.acquire()
            if self.steps_per_execution==None and self.stop_func_(g_lock):
                return None,None,None
            try:
                if hasattr(self.nn,'gradient'):
                    try:
                        gradient=self.nn.gradient(tape,loss)
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])
                else:
                    gradient=tape.gradient(loss,self.nn.param)
            except Exception as e:
                raise e
            g_lock.release()
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if self.stop_flag.value==True:
                        return None,None,None
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.steps_per_execution==None and self.stop_func_(lock[0]):
                return None,None,None
            if hasattr(self.nn,'attenuate'):
                gradient=self.nn.attenuate(gradient,p)
            try:
                try:
                    param=self.nn.opt(gradient,p)
                except Exception:
                    param=self.nn.opt(gradient)
            except Exception as e:
                raise e
            lock[0].release()
        elif self.PO==3:
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if self.stop_flag.value==True:
                        return None,None,None
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            if self.steps_per_execution==None and self.stop_func_():
                return None,None,None
            try:
                if hasattr(self.nn,'gradient'):
                    try:
                        gradient=self.nn.gradient(tape,loss)
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])
                else:
                    gradient=tape.gradient(loss,self.nn.param)
            except Exception as e:
                raise e
            if hasattr(self.nn,'attenuate'):
                gradient=self.nn.attenuate(gradient,p)
            try:
                try:
                    param=self.nn.opt(gradient,p)
                except Exception:
                    param=self.nn.opt(gradient)
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
        return
    
    
    def train7(self,train_ds,p,test_batch,lock,g_lock):
        while True:
            for data_batch,labels_batch in train_ds:
                if hasattr(self.nn,'data_func'):
                    data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)
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
                if hasattr(self.nn,'attenuate'):
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter.scatter_update(tf.IndexedSlices(0,p))
                    self.nn.opt_counter[0]=opt_counter
                output,batch_loss,param=self.opt(data_batch,labels_batch,p,lock,g_lock)
                if self.stop_flag.value==True:
                    return
                self.param[7]=param
                if self.priority_flag==True:
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')
                    opt_counter+=1
                if hasattr(self.nn,'attenuate'):
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter.assign(opt_counter+1)
                    self.nn.opt_counter[0]=opt_counter
                self.nn.bc[0]=sum(self._batch_counter)+self.bc
                _batch_counter=self._batch_counter[p]
                _batch_counter.assign_add(1)
                self._batch_counter[p]=_batch_counter
                try:
                    if hasattr(self.nn,'accuracy'):
                        try:
                            batch_acc=self.nn.accuracy(output,labels_batch,p)
                        except Exception:
                            batch_acc=self.nn.accuracy(output,labels_batch)
                except Exception as e:
                    raise e
                if hasattr(self.nn,'accuracy'):
                    self.total_loss[p]+=batch_loss
                    self.total_acc[p]+=batch_acc
                else:
                    self.total_loss[p]+=batch_loss
                self.batch_counter[p]+=1
                self.batch_counter_[p]+=1
                if self.PO==1 or self.PO==2:
                    lock[1].acquire()
                batches=np.sum(self.batch_counter_)
                if self.steps_per_execution!=None and batches%self.steps_per_execution==0:
                    loss=np.sum(self.total_loss)/self.steps_per_execution
                    if hasattr(self.nn,'accuracy'):
                        train_acc=np.sum(self.total_acc)/self.steps_per_execution
                    if self.test_flag==True:
                        if hasattr(self.nn,'accuracy'):
                            self.test_loss.value,self.test_acc.value=self.test(self.test_data,self.test_labels,test_batch)
                        else:
                            self.test_loss.value=self.test(self.test_data,self.test_labels,test_batch)
                    if self.stop_func():
                        if self.save_param_only==False:
                            self.save_()
                        else:
                            self.save_param_()
                if self.save_freq_!=None and batches%self.save_freq_==0:
                    if self.save_param_only==False:
                        self.save_()
                    else:
                        self.save_param_()
                batches=np.sum(self.batch_counter)
                if batches>=self.batches:
                    batch_counter=np.frombuffer(self.batch_counter.get_obj(),dtype='i')
                    batch_counter*=0
                    if lock!=None and type(lock)!=list:
                        lock.acquire()
                    loss=np.sum(self.total_loss)/batches
                    if hasattr(self.nn,'accuracy'):
                        train_acc=np.sum(self.total_acc)/batches
                    self.total_epoch.value+=1
                    self.train_loss.value=loss
                    self.train_loss_list.append(loss)
                    if hasattr(self.nn,'accuracy'):
                        self.train_acc.value=train_acc
                        self.train_acc_list.append(train_acc)
                    if lock!=None and type(lock)!=list:
                        lock.release()
                    if self.test_flag==True:
                        if hasattr(self.nn,'accuracy'):
                            self.test_loss.value,self.test_acc.value=self.test(self.test_data,self.test_labels,test_batch)
                            self.test_loss_list.append(self.test_loss.value)
                            self.test_acc_list.append(self.test_acc.value)
                        else:
                            self.test_loss.value=self.test(self.test_data,self.test_labels,test_batch)
                            self.test_loss_list.append(self.test_loss.value)
                    if self.save_freq_==None:
                        self.save_()
                    self.epoch_counter.value+=1
                    self.nn.ec[0]=sum(self._epoch_counter)+self.ec
                    _epoch_counter=self._epoch_counter[p]
                    _epoch_counter.assign_add(1)
                    self._epoch_counter[p]=_epoch_counter
                    total_loss=np.frombuffer(self.total_loss.get_obj(),dtype='f')
                    total_loss*=0
                    if hasattr(self.nn,'accuracy'):
                        total_acc=np.frombuffer(self.total_acc.get_obj(),dtype='f')
                        total_acc*=0
                if self.PO==1 or self.PO==2:
                    lock[1].release()
                if self.epoch!=None and self.epoch_counter.value>=self.epoch:
                    self.param[7]=param
                    return
    
    
    def train(self,p,lock=None,g_lock=None,test_batch=None):
        if self.train_dataset is not None and type(self.train_dataset)==list:
            train_ds=self.train_dataset[p]
        elif self.train_dataset is not None:
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
        if hasattr(self.nn,'counter'):
            self.nn.counter.append(0)
        while True:
            if hasattr(self.nn,'save'):
                self.nn.save(self.save,p)
            if hasattr(self.nn,'stop_flag'):
                if self.nn.stop_flag==True:
                    return
            if hasattr(self.nn,'stop_func'):
                if self.nn.stop_func(p):
                    return
            if hasattr(self.nn,'suspend_func'):
                self.nn.suspend_func(p)
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
                if hasattr(self.nn,'accuracy'):
                    try:
                        acc=self.nn.accuracy(output,data[1])
                    except Exception:
                        self.exception_list[p]=True
                    if len(self.nn.train_acc_list)==self.nn.max_length:
                        del self.nn.train_acc_list[0]
                    self.nn.train_acc_list.append(acc)
            except Exception as e:
                self.nn.exception_list[p]=e
            try:
                if hasattr(self.nn,'counter'):
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
            if hasattr(self.nn,'accuracy'):
                acc=self.nn.accuracy(output,labels)
            else:
               acc=None 
        except Exception as e:
            raise e
        return loss,acc
    
    
    def test(self,test_data=None,test_labels=None,batch=None,test_dataset=None):
        if self.process_t!=None:
            parallel_test_=parallel_test(self.nn,test_data,test_labels,self.process_t,batch,self.prefetch_batch_size_t,test_dataset)
            if type(self.test_data)!=list:
                parallel_test_.segment_data()
            processes=[]
            for p in range(self.process_t):
                process=Process(target=parallel_test_.test,args=(p,self.nn.training))
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            try:
                if hasattr(self.nn,'accuracy'):
                    test_loss,test_acc=parallel_test_.loss_acc()
                else:
                    test_loss=parallel_test_.loss_acc()
            except Exception as e:
                raise e
        elif batch!=None:
            total_loss=0
            total_acc=0
            if self.test_dataset!=None:
                batches=0
                for data_batch,labels_batch in self.test_dataset:
                    self.nn.training()
                    batches+=1
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch)
                    total_loss+=batch_loss
                    if hasattr(self.nn,'accuracy'):
                        total_acc+=batch_acc
                    self.nn.training(True)
                test_loss=total_loss.numpy()/batches
                if hasattr(self.nn,'accuracy'):
                    test_acc=total_acc.numpy()/batches
            else:
                shape0=test_data.shape[0]
                batches=int((shape0-shape0%batch)/batch)
                for j in range(batches):
                    self.nn.training()
                    index1=j*batch
                    index2=(j+1)*batch
                    data_batch=test_data[index1:index2]
                    labels_batch=test_labels[index1:index2]
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch)
                    total_loss+=batch_loss
                    if hasattr(self.nn,'accuracy'):
                        total_acc+=batch_acc
                    self.nn.training(True)
                if shape0%batch!=0:
                    self.nn.training()
                    batches+=1
                    index1=batches*batch
                    index2=batch-(shape0-batches*batch)
                    data_batch=tf.concat([test_data[index1:],test_data[:index2]],0)
                    labels_batch=tf.concat([test_labels[index1:],test_labels[:index2]],0)
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch)
                    total_loss+=batch_loss
                    if hasattr(self.nn,'accuracy'):
                        total_acc+=batch_acc
                    self.nn.training(True)
                test_loss=total_loss.numpy()/batches
                if hasattr(self.nn,'accuracy'):
                    test_acc=total_acc.numpy()/batches
        else:
            batch_loss,batch_acc=self.test_(test_data,test_labels)
            test_loss=test_loss.numpy()
            if hasattr(self.nn,'accuracy'):
                test_acc=test_acc.numpy()
        if hasattr(self.nn,'accuracy'):
            return test_loss,test_acc
        else:
            return test_loss
    
    
    def stop_func(self):
        if self.end():
            self.save(self.path)
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
    
    
    def save_(self):
        if self.path!=None and self.epoch_.value%self.save_freq==0:
            self._save()
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
        if hasattr(self.nn,'lr'):
            print('learning rate:{0}'.format(self.nn.lr))
            print()
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
        plt.plot(np.arange(1,self.total_epoch.value+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch.value+1))
        plt.show()
        print('train loss:{0:.6f}'.format(self.train_loss.value))
        if hasattr(self.nn,'accuracy'):
            plt.figure(2)
            plt.plot(np.arange(1,self.total_epoch.value+1),self.train_acc_list)
            plt.title('train acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.xticks(np.arange(1,self.total_epoch.value+1))
            plt.show()
            if self.acc_flag=='%':
                print('train acc:{0:.1f}'.format(self.train_acc.value*100))
            else:
                print('train acc:{0:.6f}'.format(self.train_acc.value)) 
        return
    
    
    def visualize_test(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch.value+1),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch.value+1))
        plt.show()
        print('test loss:{0:.6f}'.format(self.test_loss.value))
        if hasattr(self.nn,'accuracy'):
            plt.figure(2)
            plt.plot(np.arange(1,self.total_epoch.value+1),self.test_acc_list)
            plt.title('test acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.xticks(np.arange(1,self.total_epoch.value+1))
            plt.show()
            if self.acc_flag=='%':
                print('test acc:{0:.1f}'.format(self.test_acc.value*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc.value))  
        return 
    
    
    def visualize_comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch.value+1),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(1,self.total_epoch.value+1),self.test_loss_list,'r-',label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch.value+1))
        plt.show()
        print('train loss:{0:.6f}'.format(self.train_loss.value))
        if hasattr(self.nn,'accuracy'):
            plt.figure(2)
            plt.plot(np.arange(1,self.total_epoch.value+1),self.train_acc_list,'b-',label='train acc')
            if self.test_flag==True:
                plt.plot(np.arange(1,self.total_epoch.value+1),self.test_acc_list,'r-',label='test acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.xticks(np.arange(1,self.total_epoch.value+1))
            plt.show()
            if self.acc_flag=='%':
                print('train acc:{0:.1f}'.format(self.train_acc.value*100))
            else:
                print('train acc:{0:.6f}'.format(self.train_acc.value))
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
    
    
    def save_param_(self):
        if self.save_best_only==False:
            if self.save_flag.value==True:
                return
            if self.max_save_files==None or self.max_save_files==1:
                parameter_file=open(self.path,'wb')
            else:
                if hasattr(self.nn,'accuracy') and self.test_flag==True:
                    path=self.path.replace(self.path[self.path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch.value,self.train_acc.value,self.test_acc.value))
                elif hasattr(self.nn,'accuracy'):
                    path=self.path.replace(self.path[self.path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch.value,self.train_acc.value))
                else:
                    path=self.path.replace(self.path[self.path.find('.'):],'-{0}.dat'.format(self.total_epoch.value))
                parameter_file=open(path,'wb')
                self.file_list.append([path])
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0][0])
                    del self.path_list[0]
            pickle.dump(self.param[7],parameter_file)
            parameter_file.close()
        else:
            if self.monitor=='val_loss':
                if self.test_loss<self.val_loss:
                    self.val_loss=self.test_loss
                    self.save_param(self.path)
                if self.val_loss==0:
                    self.val_loss=self.test_loss
            elif self.monitor=='val_accuracy':
                if self.test_acc>self.val_accuracy:
                    self.val_accuracy=self.test_acc
                    self.save_param(self.path)
                if self.val_accuracy==1:
                    self.val_accuracy=self.test_acc
        return
    
    
    def save_param(self,path):
        parameter_file=open(path,'wb')
        pickle.dump(self.param[7],parameter_file)
        parameter_file.close()
        return
    
    
    def restore_param(self,path):
        parameter_file=open(path,'rb')
        param=pickle.load(parameter_file)
        param_flat=nest.flatten(param)
        param_flat_=nest.flatten(self.nn.param)
        for i in range(len(param_flat)):
            state_ops.assign(param_flat_[i],param_flat[i])
        self.nn.param=nest.pack_sequence_as(self.nn.param,param_flat_)
        parameter_file.close()
        return
    
    
    def _save(self):
        if self.save_best_only==False:
            if self.save_flag.value==True:
                return
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(self.path,'wb')
            else:
                if hasattr(self.nn,'accuracy') and self.test_flag==True:
                    path=self.path.replace(self.path[self.path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch.value,self.train_acc.value,self.test_acc.value))
                elif hasattr(self.nn,'accuracy'):
                    path=self.path.replace(self.path[self.path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch.value,self.train_acc.value))
                else:
                    path=self.path.replace(self.path[self.path.find('.'):],'-{0}.dat'.format(self.total_epoch.value))
                output_file=open(path,'wb')
                self.file_list.append([path])
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0][0])
                    del self.path_list[0]
            self.update_nn_param()
            self.nn.opt_counter=None
            self.nn.ec=self.nn.ec[0]
            self.nn.bc=self.nn.bc[0]
            self._epoch_counter=list(self._epoch_counter)
            self._batch_counter=list(self._batch_counter)
            self.nn.optimizer.convert_to_list()
            Module.convert_to_list()
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
            pickle.dump(list(self.train_loss_list),output_file)
            pickle.dump(list(self.train_acc_list),output_file)
            pickle.dump(self.test_flag,output_file)
            if self.test_flag==True:
                pickle.dump(self.test_loss.value,output_file)
                pickle.dump(self.test_acc.value,output_file)
                pickle.dump(list(self.test_loss_list),output_file)
                pickle.dump(list(self.test_acc_list),output_file)
            pickle.dump(self.total_epoch.value,output_file)
            output_file.close()
        else:
            if self.monitor=='val_loss':
                if self.test_loss<self.val_loss:
                    self.val_loss=self.test_loss
                    self.save(self.path)
                if self.val_loss==0:
                    self.val_loss=self.test_loss
            elif self.monitor=='val_accuracy':
                if self.test_acc>self.val_accuracy:
                    self.val_accuracy=self.test_acc
                    self.save(self.path)
                if self.val_accuracy==1:
                    self.val_accuracy=self.test_acc
        return
    
    
    def save(self,path):
        output_file=open(path,'wb')
        self.update_nn_param()
        self.nn.opt_counter=None
        self.nn.ec=self.nn.ec[0]
        self.nn.bc=self.nn.bc[0]
        self._epoch_counter=list(self._epoch_counter)
        self._batch_counter=list(self._batch_counter)
        self.nn.optimizer.convert_to_list()
        Module.convert_to_list()
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
        pickle.dump(list(self.train_loss_list),output_file)
        pickle.dump(list(self.train_acc_list),output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss.value,output_file)
            pickle.dump(self.test_acc.value,output_file)
            pickle.dump(list(self.test_loss_list),output_file)
            pickle.dump(list(self.test_acc_list),output_file)
        pickle.dump(self.total_epoch.value,output_file)
        output_file.close()
        return
    
	
    def restore(self,s_path,manager):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        self.convert_to_shared_list(manager)
        Module.convert_to_shared_list(manager)
        self.nn.km=1
        self.nn.opt_counter=manager.list([tf.Variable(tf.zeros([self.process]))])
        self.ec=self.nn.ec
        self.bc=self.nn.bc
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
        self.train_loss_list[:]=pickle.load(input_file)
        self.train_acc_list[:]=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss.value=pickle.load(input_file)
            self.test_acc.value=pickle.load(input_file)
            self.test_loss_list[:]=pickle.load(input_file)
            self.test_acc_list[:]=pickle.load(input_file)
        self.total_epoch.value=pickle.load(input_file)
        input_file.close()
        return
