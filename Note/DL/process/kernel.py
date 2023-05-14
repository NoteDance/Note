import tensorflow as tf
from tensorflow import function
from multiprocessing import Value,Array
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.PO=None
        self.process=None
        self.train_ds=None
        self.data_segment_flag=False
        self.batches=None
        self.buffer_size=None
        self.epoch_=None
        self.epoch_counter=0
        self.stop=False
        self.stop_flag=False
        self.save_flag=False
        self.save_epoch=None
        self.batch=None
        self.epoch=0
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.acc_flag='%'
        self.train_counter=0
        self.opt_counter=None
        self.muti_p=None
        self.muti_s=None
        self.muti_save=1
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
    
    
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        self.train_data=train_data
        self.train_labels=train_labels
        self.train_dataset=train_dataset
        if self.data_segment_flag==True:
            self.train_data,self.train_labels=self.segment_data()
        if type(train_data)==list:
            self.data_batch=[x for x in range(len(train_data))]
        if type(train_labels)==list:
            self.labels_batch=[x for x in range(len(train_labels))]
        self.test_data=test_data
        self.test_labels=test_labels
        self.test_dataset=test_dataset
        try:
            if test_data==None:
                self.test_flag=False
        except ValueError:
            self.test_flag=True
        self.process_num=np.arange(self.process)
        self.process_num=list(self.process_num)
        self.batch_counter=np.zeros(self.process,dtype=np.int32)
        self.total_loss=np.zeros(self.process,dtype=np.float32)
        if self.train_dataset==None:
            if type(self.train_data)==list:
                self.shape0=train_data[0].shape[0]
            else:
                self.shape0=train_data.shape[0]
        return
    
    
    def segment_data(self):
        if len(self.train_data)!=self.process:
            data=None
            labels=None
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
                
    
    def init(self,manager,batches):
        self.epoch_counter=Value('i',0)
        self.process_num=manager.list(self.process_num)
        self.batch_counter=Array('i',self.batch_counter)
        self.total_loss=Array('f',self.total_loss)
        self.total_epoch=Value('i',0)
        self.train_loss=Value('f',0)
        self.train_loss_list=manager.list([])
        self.batches=Value('i',batches)
        self.stop_flag=Value('b',0)
        self.save_flag=Value('b',0)
        self.param=manager.dict()
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
                        data_batch[i]=tf.concat([self.train_data[i][index1:],self.train_data[i][:index2]],0)
                else:
                    data_batch=tf.concat([self.train_data[index1:],self.train_data[:index2]],0)
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=tf.concat([self.train_labels[i][index1:],self.train_labels[i][:index2]],0)
                else:
                    labels_batch=tf.concat([self.train_labels[index1:],self.train_labels[:index2]],0)
            except:
                if type(self.train_data)==list:
                    for i in range(len(self.train_data)):
                        data_batch[i]=np.concatenate([self.train_data[i][index1:],self.train_data[i][:index2]],0)
                else:
                    data_batch=np.concatenate([self.train_data[index1:],self.train_data[:index2]],0)
                if type(self.train_labels)==list:
                    for i in range(len(self.train_data)):
                        labels_batch[i]=np.concatenate([self.train_labels[i][index1:],self.train_labels[i][:index2]],0)
                else:
                    labels_batch=np.concatenate([self.train_labels[index1:],self.train_labels[:index2]],0)
        return data_batch,labels_batch
    
    
    @function(jit_compile=True)
    def tf_opt_t(self,data,labels,t=None,ln=None,lock=None):
        try:
            if self.nn.GradientTape!=None:
                tape,output,loss=self.nn.GradientTape(data,labels,t)
        except AttributeError:
            with tf.GradientTape(persistent=True) as tape:
                try:
                    try:
                        output=self.nn.fp(data)
                        loss=self.nn.loss(output,labels)
                    except TypeError:
                        output,loss=self.nn.fp(data,labels)
                except TypeError:
                    try:
                        output=self.nn.fp(data,t)
                        loss=self.nn.loss(output,labels)
                    except TypeError:
                        output,loss=self.nn.fp(data,labels,t)
        try:
            if self.nn.attenuate!=None:
                self.opt_counter[t]=0
        except AttributeError:
            pass
        if self.PO==1:
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return None,0
            try:
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                gradient=tape.gradient(loss,self.nn.param)
            try:
                if self.nn.attenuate!=None:
                    gradient=self.nn.attenuate(gradient,self.opt_counter,t)
            except AttributeError:
                pass
            try:
                param=self.nn.opt(gradient)
            except TypeError:
                self.nn.opt(gradient,t)
            try:
                if self.nn.attenuate!=None:
                    for i in range(len(self.opt_counter)):
                        self.opt_counter[i]+=1
            except AttributeError:
                pass
            lock[0].release()
        elif self.PO==2:
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return None,0
            try:
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                gradient=tape.gradient(loss,self.nn.param)
            lock[0].release()
            lock[1].acquire()
            try:
                if self.nn.attenuate!=None:
                    gradient=self.nn.attenuate(gradient,self.opt_counter,t)
            except AttributeError:
                pass
            try:
                self.nn.opt(gradient)
            except TypeError:
                self.nn.opt(gradient,t)
            try:
                if self.nn.attenuate!=None:
                    for i in range(len(self.opt_counter)):
                        self.opt_counter[i]+=1
            except AttributeError:
                pass
            lock[1].release()
        return output,loss,param
    
    
    def opt_t(self,data,labels,t=None,lock=None):
        output,loss,param=self.tf_opt_t(data,labels,int(t),lock=lock)
        return output,loss,param
    
    
    def train7(self,train_ds,t,test_batch,lock):
        while True:
            for data_batch,labels_batch in train_ds:
                output,batch_loss,weight=self.opt_t(data_batch,labels_batch,t,lock=lock)
                try:
                    self.nn.bc[t]+=1
                except AttributeError:
                    pass
                try:
                    if self.nn.accuracy!=None:
                        batch_acc=self.nn.accuracy(output,labels_batch)
                except AttributeError:
                    pass
                try:
                    if self.nn.accuracy!=None:
                        self.total_loss[t]+=batch_loss
                        self.total_acc[t]+=batch_acc
                except AttributeError:
                    self.total_loss[t]+=batch_loss
                self.batch_counter[t]+=1
                if self.PO==1:
                    lock[1].acquire()
                else:
                    lock[2].acquire()
                batches=np.sum(self.batch_counter)
                if batches>=self.batches.value:
                    for i in range(len(self.batch_counter)):
                        self.batch_counter[i]=0
                    loss=np.sum(self.total_loss)/batches
                    try:
                        if self.nn.accuracy!=None:
                            train_acc=np.sum(self.total_acc)/batches
                    except AttributeError:
                        pass
                    self.total_epoch.value+=1
                    self.train_loss.value=loss
                    self.train_loss_list.append(loss)
                    try:
                        if self.nn.accuracy!=None:
                            self.train_acc.value=train_acc
                            self.train_acc_list.append(train_acc)
                    except AttributeError:
                        pass
                    if self.test_flag==True:
                        if self.process_thread_t==None:
                            self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch,t)
                        else:
                            self.test_loss,self.test_acc=self.test(batch=test_batch)
                        self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                    self.print_save()
                    self.epoch_counter.value+=1
                    try:
                        self.nn.bc[t]=0
                    except AttributeError:
                        pass
                    try:
                        self.nn.ec+=1
                    except AttributeError:
                        pass
                    for i in range(len(self.total_loss)):
                        self.total_loss[i]=0
                    try:
                        if self.nn.accuracy!=None:
                            for i in range(len(self.total_acc)):
                                self.total_acc[i]=0
                    except AttributeError:
                        pass
                if self.PO==1:
                    lock[1].release()
                else:
                    lock[2].release()
                if self.epoch_counter.value==self.epoch_:
                    self.param[7]=weight
                    return
    
    
    def train(self,batch=None,lock=None,test_batch=None):
        if self.process!=None:
            if self.PO==1:
                lock[1].acquire()
            else:
                lock[2].acquire()
            t=self.process_num.pop(0)
            if self.PO==1:
                lock[1].release()
            else:
                lock[2].release()
        self.batch=batch
        self.train_counter+=1
        if self.epoch_!=None:
            if self.train_dataset!=None:
                train_ds=self.train_dataset
            else:
                if self.data_segment_flag==True:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data[t],self.train_labels[t])).batch(batch)
                elif self.buffer_size!=None:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).shuffle(self.buffer_size).batch(batch)
                else:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).batch(batch)
        self.train7(train_ds,t,test_batch,lock)
        return
    
    
    def test(self,test_data=None,test_labels=None,batch=None,param=None,t=None):
        if type(test_data)==list:
            data_batch=[x for x in range(len(test_data))]
        if type(test_labels)==list:
            labels_batch=[x for x in range(len(test_labels))]
        if batch!=None:
            total_loss=0
            total_acc=0
            if self.test_dataset!=None:
                for data_batch,labels_batch in self.test_dataset:
                    if self.process==None or t==None:
                        if param==None:
                            output=self.nn.fp(data_batch)
                        else:
                            output=self.nn.fp(data_batch,param)
                    else:
                        if param==None:
                            output=self.nn.fp(data_batch,t)
                        else:
                            output=self.nn.fp(data_batch,param,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                    total_loss+=batch_loss
                    try:
                        if self.nn.accuracy!=None:
                            batch_acc=self.nn.accuracy(output,labels_batch)
                            total_acc+=batch_acc
                    except AttributeError:
                        pass
            else:
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
                    if self.process==None or t==None:
                        if param==None:
                            output=self.nn.fp(data_batch)
                        else:
                            output=self.nn.fp(data_batch,param)
                    else:
                        if param==None:
                            output=self.nn.fp(data_batch,t)
                        else:
                            output=self.nn.fp(data_batch,param,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                    total_loss+=batch_loss
                    try:
                        if self.nn.accuracy!=None:
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
                                data_batch[i]=tf.concat([test_data[i][index1:],test_data[i][:index2]],0)
                        else:
                            data_batch=tf.concat([test_data[index1:],test_data[:index2]],0)
                        if type(self.test_labels)==list:
                            for i in range(len(test_labels)):
                                labels_batch[i]=tf.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                        else:
                            labels_batch=tf.concat([test_labels[index1:],test_labels[:index2]],0)
                    except:
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
                    if self.process==None or t==None:
                        if param==None:
                            output=self.nn.fp(data_batch)
                        else:
                            output=self.nn.fp(data_batch,param)
                    else:
                        if param==None:
                            output=self.nn.fp(data_batch,t)
                        else:
                            output=self.nn.fp(data_batch,param,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                    total_loss+=batch_loss
                    try:
                        if self.nn.accuracy!=None:
                            batch_acc=self.nn.accuracy(output,labels_batch)
                            total_acc+=batch_acc
                    except AttributeError:
                        pass
            test_loss=total_loss.numpy()/batches
            try:
                if self.nn.accuracy!=None:
                    test_acc=total_acc.numpy()/batches
            except AttributeError:
                pass
        else:
            if self.process==None or t==None:
                if self.param==None:
                    output=self.nn.fp(test_data)
                else:
                    output=self.nn.fp(test_data,param)
            else:
                if self.param==None:
                    output=self.nn.fp(test_data,t)
                else:
                    output=self.nn.fp(test_data,param,t)
            test_loss=self.nn.loss(output,test_labels)
            test_loss=test_loss.numpy()
            try:
                if self.nn.accuracy!=None:
                    test_acc=self.nn.accuracy(output,test_labels)
                    test_acc=test_acc.numpy()
            except AttributeError:
                pass
        try:
            if self.nn.accuracy!=None:
                return test_loss,test_acc
        except AttributeError:
            return test_loss,None
    
    
    def stop_func(self):
        if self.end():
            self.save(self.total_epoch.value,True)
            self.save_flag.value=True
            self.stop_flag.value=True
            return True
        return False
    
    
    def stop_func_(self,lock):
        if self.stop==True:
            if self.stop_flag.value==True or self.stop_func():
                lock.release()
                return True
    
    
    def print_save(self):
        if self.epoch_!=None:
            if self.muti_p!=None:
                muti_p=self.muti_p-1
                if self.epoch_%10!=0:
                    p=self.epoch_-self.epoch_%muti_p
                    p=int(p/muti_p)
                    if p==0:
                        p=1
                else:
                    p=self.epoch_/(muti_p+1)
                    p=int(p)
                    if p==0:
                        p=1
                if self.epoch%p==0:
                    if self.test_flag==False:
                        try:
                            if self.nn.accuracy!=None:
                                print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch.value,self.train_loss.value))
                                if self.acc_flag=='%':
                                    print('epoch:{0}   accuracy:{1:.1f}'.format(self.total_epoch.value,self.train_acc.value*100))
                                else:
                                    print('epoch:{0}   accuracy:{1:.6f}'.format(self.total_epoch.value,self.train_acc.value))
                                print()
                        except AttributeError:
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
                        except AttributeError:   
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(self.total_epoch,self.train_loss.value,self.test_loss.value))
                            print()
            if self.muti_s!=None:
                muti_s=self.muti_s-1
                if self.epoch_%10!=0:
                    s=self.epoch_-self.epoch_%muti_s
                    s=int(s/muti_s)
                    if s==0:
                        s=1
                else:
                    s=self.epoch_/(muti_s+1)
                    s=int(s)
                    if s==0:
                        s=1
                if self.muti_save!=None and self.epoch%s==0:
                    if self.muti_save==1:
                        self.save(self.total_epoch.value)
                    else:
                        self.save(self.total_epoch.value,False)
            self.epoch+=1
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
        except AttributeError:
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
        except AttributeError:
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
        except AttributeError:
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
        except AttributeError:
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
        pickle.dump(self.nn.param,parameter_file)
        parameter_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if self.save_flag==True:
            return
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
        except:
            opt=self.nn.opt
            self.nn.opt=None
            pickle.dump(self.nn,output_file)
            self.nn.opt=opt
        try:
            pickle.dump(tf.keras.optimizers.serialize(opt),output_file)
        except:
            try:
                pickle.dump(self.nn.serialize(),output_file)
            except:
                pickle.dump(None,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.end_acc,output_file)
        pickle.dump(self.end_test_loss,output_file)
        pickle.dump(self.end_test_acc,output_file)
        pickle.dump(self.acc_flag,output_file)
        pickle.dump(self.file_list,output_file)
        pickle.dump(self.train_counter,output_file)
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
        output_file.close()
        if self.save_flag==True:
            print('\nSystem have stopped,Neural network have saved.')
        return
    
	
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        try:
            self.nn.km=1
        except AttributeError:
            pass
        opt_serialized=pickle.load(input_file)
        try:
            self.nn.opt=tf.keras.optimizers.deserialize(opt_serialized)
        except:
            try:
                self.nn.deserialize(opt_serialized)
            except:
                pass
        self.batch=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.end_acc=pickle.load(input_file)
        self.end_test_loss=pickle.load(input_file)
        self.end_test_acc=pickle.load(input_file)
        self.acc_flag=pickle.load(input_file)
        self.file_list=pickle.load(input_file)
        self.train_counter=pickle.load(input_file)
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
        input_file.close()
        return
