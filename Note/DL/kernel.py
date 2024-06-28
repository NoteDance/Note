from tensorflow import function
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Process
import numpy as np
from Note.DL.dl.test import parallel_test
import matplotlib.pyplot as plt
import pickle
import os
import time


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        if hasattr(self.nn,'km'):
            self.nn.km=1
        self.platform=None
        self.batches=None
        self.process_t=None
        self.prefetch_batch_size_t=None
        self.suspend=False
        self.stop=False
        self.stop_flag=False
        self.save_epoch=None
        self.batch=None
        self.epoch=0
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.acc_flag='%'
        self.train_counter=0
        self.filename='save.dat'
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
    
    
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        self.train_data=train_data
        self.train_labels=train_labels
        self.train_dataset=train_dataset
        self.test_data=test_data
        self.test_labels=test_labels
        self.test_dataset=test_dataset
        if test_data is not None or test_dataset is not None:
            self.test_flag=True
        if train_data is not None:
            self.shape0=train_data.shape[0]
        return
    
    
    def init(self):
        self.suspend=False
        self.stop=False
        self.stop_flag=False
        self.save_epoch=None
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.train_loss=None
        self.train_acc=None
        self.test_loss=None
        self.test_acc=None
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
    
    
    def end(self):
        if self.end_acc!=None and self.train_acc!=None and self.train_acc>self.end_acc:
            return True
        elif self.end_loss!=None and self.train_loss!=None and self.train_loss<self.end_loss:
            return True
        elif self.end_test_acc!=None and self.test_acc!=None and self.test_acc>self.end_test_acc:
            return True
        elif self.end_test_loss!=None and self.test_loss!=None and self.test_loss<self.end_test_loss:
            return True
        elif self.end_acc!=None and self.end_test_acc!=None:
            if self.train_acc!=None and self.test_acc!=None and self.train_acc>self.end_acc and self.test_acc>self.end_test_acc:
                return True
        elif self.end_loss!=None and self.end_test_loss!=None:
            if self.train_loss!=None and self.test_loss!=None and self.train_loss<self.end_loss and self.test_loss<self.end_test_loss:
                return True
    
    
    def loss_acc(self,output=None,labels_batch=None,loss=None,test_batch=None,total_loss=None,total_acc=None):
        if self.batch!=None:
            total_loss+=loss
            if hasattr(self.nn,'accuracy'):
                batch_acc=self.nn.accuracy(output,labels_batch)
                total_acc+=batch_acc
            return total_loss,total_acc
        else:
            loss=loss.numpy()
            self.train_loss=loss
            self.train_loss_list.append(loss)
            if hasattr(self.nn,'accuracy'):
                acc=self.nn.accuracy(output,self.train_labels)
                acc=acc.numpy()
                self.train_acc=acc
                self.train_acc_list.append(acc)
            if self.test_flag==True:
                if hasattr(self.nn,'accuracy'):
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
                    self.test_acc_list.append(self.test_acc)
                else:
                    self.test_loss=self.test(self.test_data,self.test_labels,test_batch)
                    self.test_loss_list.append(self.test_loss)
            return
    
    
    def data_func(self,batch=None,index1=None,index2=None,j=None,flag=None):
        if flag==None:
            if batch!=1:
                data_batch=self.train_data[index1:index2]
            else:
                data_batch=self.train_data[j]
            if batch!=1:
                labels_batch=self.train_labels[index1:index2]
            else:
                labels_batch=self.train_labels[j]
        else:
            try:
                try:
                    data_batch=self.platform.concat([self.train_data[index1:],self.train_data[:index2]],0)
                    labels_batch=self.platform.concat([self.train_labels[index1:],self.train_labels[:index2]],0)
                except Exception:
                    data_batch=np.concatenate([self.train_data[index1:],self.train_data[:index2]],0)
                    labels_batch=np.concatenate([self.train_labels[index1:],self.train_labels[:index2]],0)
            except Exception as e:
                raise e
        return data_batch,labels_batch
    
    
    @function(jit_compile=True)
    def tf_opt(self,data,labels):
        try:
            if hasattr(self.nn,'GradientTape'):
                tape,output,loss=self.nn.GradientTape(data,labels)
            else:
                with self.platform.GradientTape(persistent=True) as tape:
                    try:
                        output=self.nn.fp(data)
                        loss=self.nn.loss(output,labels)
                    except Exception:
                        output,loss=self.nn.fp(data,labels)
        except Exception as e:
            raise e
        if hasattr(self.nn,'gradient'):
            gradient=self.nn.gradient(tape,loss)
        else:
            gradient=tape.gradient(loss,self.nn.param)
        if hasattr(self.nn.opt,'apply_gradients'):
            self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
        else:
            self.nn.opt(gradient)
        return output,loss
    
    
    def pytorch_opt(self,data,labels):
        output=self.nn.fp(data)
        loss=self.nn.loss(output,labels)
        if hasattr(self.nn.opt,'zero_grad'):
            self.nn.opt.zero_grad()
            loss.backward()
            self.nn.opt.step()
        else:
            self.nn.opt(loss)
        return output,loss
    
    
    def opt(self,data,labels):
        if hasattr(self.platform,'DType'):
            output,loss=self.tf_opt(data,labels)
        else:
            output,loss=self.pytorch_opt(data,labels)
        return output,loss
    
    
    def _train(self,batch=None,test_batch=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            if self.train_dataset!=None:
                for data_batch,labels_batch in self.train_dataset:
                    if self.stop==True:
                        if self.stop_func():
                            return
                    if hasattr(self.nn,'data_func'):
                        data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)
                    output,batch_loss=self.opt(data_batch,labels_batch)
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc)
            else:
                total_loss=0
                total_acc=0
                batches=int((self.shape0-self.shape0%batch)/batch)
                for j in range(batches):
                    if self.stop==True:
                        if self.stop_func():
                            return
                    index1=j*batch
                    index2=(j+1)*batch
                    data_batch,labels_batch=self.data_func(batch,index1,index2,j)
                    if hasattr(self.nn,'data_func'):
                        data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)
                    output,batch_loss=self.opt(data_batch,labels_batch)
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc)
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
                if self.shape0%batch!=0:
                    if self.stop==True:
                        if self.stop_func():
                            return
                    batches+=1
                    index1=batches*batch
                    index2=batch-(self.shape0-batches*batch)
                    data_batch,labels_batch=self.data_func(batch,index1,index2,flag=True)
                    if hasattr(self.nn,'data_func'):
                        data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)
                    output,batch_loss=self.opt(data_batch,labels_batch)
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc)
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
            if hasattr(self.platform,'DType'):
                loss=total_loss.numpy()/batches
            else:
                loss=total_loss.detach().numpy()/batches
            if hasattr(self.nn,'accuracy'):
                train_acc=total_acc.numpy()/batches
            self.train_loss=loss
            self.train_loss_list.append(loss)
            if hasattr(self.nn,'accuracy'):
                self.train_acc=train_acc
                self.train_acc_list.append(train_acc)
            if self.test_flag==True:
                if hasattr(self.nn,'accuracy'):
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch,self.test_dataset)
                    self.test_loss_list.append(self.test_loss)
                    self.test_acc_list.append(self.test_acc)
                else:
                    self.test_loss=self.test(self.test_data,self.test_labels,test_batch,self.test_dataset)
                    self.test_loss_list.append(self.test_loss)
        else:
            output,train_loss=self.opt(self.train_data,self.train_labels)
            self.loss_acc(output=output,labels_batch=labels_batch,loss=train_loss,test_batch=test_batch,total_loss=total_loss,total_acc=total_acc)
        return
    
    
    def train(self,batch=None,epoch=None,test_batch=None,save=False,one=True,p=None,s=None):
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
        if epoch!=None:
            for i in range(epoch):
                t1=time.time()
                self._train(batch,test_batch)
                if self.stop_flag==True:
                    return
                if hasattr(self.nn,'ec'):
                    try:
                        self.nn.ec.assign_add(1)
                    except Exception:
                        self.nn.ec+=1
                self.total_epoch+=1
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
                    if self.test_flag==False:
                        if hasattr(self.nn,'accuracy'):
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            if self.acc_flag=='%':
                                print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))
                            else:
                                print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            print()
                    else:
                        if hasattr(self.nn,'accuracy'):
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                            if self.acc_flag=='%':
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))
                            else:
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                            print()
                if save!=False and i%s==0:
                    self.save(self.total_epoch,one)
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                self._train(test_batch=test_batch)
                if self.stop_flag==True:
                    return
                i+=1
                if hasattr(self.nn,'ec'):
                    try:
                        self.nn.ec.assign_add(1)
                    except Exception:
                        self.nn.ec+=1
                self.total_epoch+=1
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
                    if self.test_flag==False:
                        if hasattr(self.nn,'accuracy'):
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            if self.acc_flag=='%':
                                print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))
                            else:
                                print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                            print()
                    else:
                        if hasattr(self.nn,'accuracy'):
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                            if self.acc_flag=='%':
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))
                            else:
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                            print()
                if save!=False and i%s==0:
                    self.save(self.total_epoch,one)
                t2=time.time()
                self.time+=(t2-t1)
        self._time=self.time-int(self.time)
        if self._time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        if self.test_flag==False:
            print('last loss:{0:.6f}'.format(self.train_loss))
        else:
            print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))
        if hasattr(self.nn,'accuracy'):
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
        print()
        print('time:{0}s'.format(self.time))
        self.training_flag=False
        return
    
    
    def train_online(self):
        while True:
            if hasattr(self.nn,'save'):
                self.nn.save(self.save)
            if hasattr(self.nn,'stop_flag'):
                if self.nn.stop_flag==True:
                    return
            if hasattr(self.nn,'stop_func'):
                if self.nn.stop_func():
                    return
            if hasattr(self.nn,'suspend_func'):
                self.nn.suspend_func()
            data=self.nn.online()
            if data=='stop':
                return
            elif data=='suspend':
                self.nn.suspend_func()
            output,loss=self.opt(data[0],data[1])
            loss=loss.numpy()
            if len(self.nn.train_loss_list)==self.nn.max_length:
                del self.nn.train_loss_list[0]
            self.nn.train_loss_list.append(loss)
            try:
                if hasattr(self.nn,'accuracy'):
                    train_acc=self.nn.accuracy(output,data[1])
                    if len(self.nn.train_acc_list)==self.nn.max_length:
                        del self.nn.train_acc_list[0]
                    self.train_acc_list.append(train_acc)
            except Exception as e:
                raise e
            if hasattr(self.nn,'counter'):
                self.nn.counter+=1
        return
    
    
    @function(jit_compile=True)
    def test_tf(self,data,labels):
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
    
    
    def test_pytorch(self,data,labels):
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
            if self.prefetch_batch_size_t==None:
                parallel_test_=parallel_test(self.nn,test_data,test_labels,self.process_t,batch,test_dataset=test_dataset)
            else:
                parallel_test_=parallel_test(self.nn,test_data,test_labels,self.process_t,batch,self.prefetch_batch_size_t,test_dataset)
            if type(self.test_data)!=list:
                parallel_test_.segment_data()
            processes=[]
            for p in range(self.process_t):
                process=Process(target=parallel_test_.test,args=(p,))
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
                    batches+=1
                    if hasattr(self.platform,'DType'):
                        batch_loss,batch_acc=self.test_tf(data_batch,labels_batch)
                    else:
                        batch_loss,batch_acc=self.test_pytorch(data_batch,labels_batch)
                    total_loss+=batch_loss
                    try:
                        total_acc+=batch_acc
                    except Exception:
                        pass
                test_loss=total_loss.numpy()/batches
                if hasattr(self.nn,'accuracy'):
                    test_acc=total_acc.numpy()/batches
            else:
                shape0=test_data.shape[0]
                batches=int((shape0-shape0%batch)/batch)
                for j in range(batches):
                    index1=j*batch
                    index2=(j+1)*batch
                    data_batch=test_data[index1:index2]
                    labels_batch=test_labels[index1:index2]
                    if hasattr(self.platform,'DType'):
                        batch_loss,batch_acc=self.test_tf(data_batch,labels_batch)
                    else:
                        batch_loss,batch_acc=self.test_pytorch(data_batch,labels_batch)
                    total_loss+=batch_loss
                    if hasattr(self.nn,'accuracy'):
                        total_acc+=batch_acc
                if shape0%batch!=0:
                    batches+=1
                    index1=batches*batch
                    index2=batch-(shape0-batches*batch)
                    try:
                        try:
                            data_batch=self.platform.concat([test_data[index1:],test_data[:index2]],0)
                            labels_batch=self.platform.concat([test_labels[index1:],test_labels[:index2]],0)
                        except Exception:
                            data_batch=np.concatenate([test_data[index1:],test_data[:index2]],0)
                            labels_batch=np.concatenate([test_labels[index1:],test_labels[:index2]],0)
                    except Exception as e:
                        raise e
                    if hasattr(self.platform,'DType'):
                        batch_loss,batch_acc=self.test_tf(data_batch,labels_batch)
                    else:
                        batch_loss,batch_acc=self.test_pytorch(data_batch,labels_batch)
                    total_loss+=batch_loss
                    if hasattr(self.nn,'accuracy'):
                        total_acc+=batch_acc
                test_loss=total_loss.numpy()/batches
                if hasattr(self.nn,'accuracy'):
                    test_acc=total_acc.numpy()/batches
        else:
            if hasattr(self.platform,'DType'):
                batch_loss,batch_acc=self.test_tf(test_data,test_labels)
            else:
                batch_loss,batch_acc=self.test_pytorch(test_data,test_labels)
            test_loss=test_loss.numpy()
            if hasattr(self.nn,'accuracy'):
                test_acc=test_acc.numpy()
        if hasattr(self.nn,'accuracy'):
            return test_loss,test_acc
        else:
            return test_loss
    
    
    def suspend_func(self):
        if self.suspend==True:
            if self.save_epoch==None:
                print('Training have suspended.')
            else:
                self._save()
            while True:
                if self.suspend==False:
                    print('Training have continued.')
                    break
        return
    
    
    def stop_func(self):
        if self.end():
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
            if hasattr(self.nn,'accuracy'):
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
            print()
            print('time:{0}s'.format(self.total_time))
            self.stop_flag=True
            return True
        return False
    
    
    def stop_func_(self):
        if self.stop==True:
            if self.stop_flag==True or self.stop_func():
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
        if hasattr(self.nn,'lr'):
            print('learning rate:{0}'.format(self.nn.lr))
            print()
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


    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        print('train loss:{0:.6f}'.format(self.train_loss))
        if hasattr(self.nn,'accuracy'):
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(1,self.total_epoch+1),self.train_acc_list)
                plt.title('train acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.xticks(np.arange(1,self.total_epoch+1))
                plt.show()
                if self.acc_flag=='%':
                    print('train acc:{0:.1f}'.format(self.train_acc*100))
                else:
                    print('train acc:{0:.6f}'.format(self.train_acc)) 
        return
    
    
    def visualize_test(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch+1),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        print('test loss:{0:.6f}'.format(self.test_loss))
        if hasattr(self.nn,'accuracy'):
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(1,self.total_epoch+1),self.test_acc_list)
                plt.title('test acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.xticks(np.arange(1,self.total_epoch+1))
                plt.show()
                if self.acc_flag=='%':
                    print('test acc:{0:.1f}'.format(self.test_acc*100))
                else:
                    print('test acc:{0:.6f}'.format(self.test_acc))  
        return 
    
    
    def visualize_comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch+1),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(1,self.total_epoch+1),self.test_loss_list,'r-',label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        print('train loss:{0:.6f}'.format(self.train_loss))
        if hasattr(self.nn,'accuracy'):
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(1,self.total_epoch+1),self.train_acc_list,'b-',label='train acc')
                if self.test_flag==True:
                    plt.plot(np.arange(1,self.total_epoch+1),self.test_acc_list,'r-',label='test acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.xticks(np.arange(1,self.total_epoch+1))
                plt.show()
                if self.acc_flag=='%':
                    print('train acc:{0:.1f}'.format(self.train_acc*100))
                else:
                    print('train acc:{0:.6f}'.format(self.train_acc))
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
    
    
    def save_param(self,path):
        parameter_file=open(path,'wb')
        pickle.dump(self.nn.param,parameter_file)
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
    
    
    def save(self,i=None,one=True):
        if one==True:
            output_file=open(self.filename,'wb')
        else:
            filename=self.filename.replace(self.filename[self.filename.find('.'):],'-{0}.dat'.format(i))
            output_file=open(filename,'wb')
            self.file_list.append([filename])
            if len(self.file_list)>self.s+1:
                os.remove(self.file_list[0][0])
                del self.file_list[0]
        try:
            pickle.dump(self.nn,output_file)
        except Exception as e:
            first_exception=e
            try:
                opt=self.nn.opt
                self.nn.opt=None
                pickle.dump(self.nn,output_file)
                self.nn.opt=opt
            except Exception as e:
                raise e
                raise first_exception
        try:
            try:
                pickle.dump(self.platform.keras.optimizers.serialize(opt),output_file)
            except Exception:
                try:
                    pickle.dump(self.nn.serialize(),output_file)
                except Exception:
                    pickle.dump(None,output_file)
        except Exception as e:
            raise e
        pickle.dump(self.batch,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.end_acc,output_file)
        pickle.dump(self.end_test_loss,output_file)
        pickle.dump(self.end_test_acc,output_file)
        pickle.dump(self.acc_flag,output_file)
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
        if hasattr(self.nn,'km'):
            self.nn.km=1
        opt_serialized=pickle.load(input_file)
        try:
            self.nn.opt=self.platform.keras.optimizers.deserialize(opt_serialized)
        except Exception:
            try:
                self.nn.deserialize(opt_serialized)
            except Exception:
                pass
        self.batch=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.end_acc=pickle.load(input_file)
        self.end_test_loss=pickle.load(input_file)
        self.end_test_acc=pickle.load(input_file)
        self.acc_flag=pickle.load(input_file)
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
