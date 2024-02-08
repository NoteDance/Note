import torch
from multiprocessing import Value,Array
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        self.nn.km=1
        self.process=None
        self.process_t=None
        self.train_ds=None
        self.prefetch_factor=None
        self.num_workers=None
        self.batches_t=None
        self.shuffle=False
        self.priority_flag=False
        self.max_opt=None
        self.epoch=None
        self.stop=False
        self.save_epoch=None
        self.batch=None
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.acc_flag='%'
        self.s=None
        self.saving_one=True
        self.filename='save.dat'
        self.test_flag=False
    
    
    def data(self,train_dataset=None,test_dataset=None):
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        if test_dataset is not None:
            self.test_flag=True
        self.batch_counter=np.zeros(self.process,dtype='int32')
        self.total_loss=np.zeros(self.process,dtype='float32')
        if hasattr(self.nn,'accuracy'):
            self.total_acc=np.zeros(self.process,dtype='float32')
        if self.priority_flag==True:
            self.opt_counter=np.zeros(self.process,dtype=np.int32)
        return
                    
    
    def init(self,manager):
        self.epoch_counter=Value('i',0)
        self.batch_counter=Array('i',self.batch_counter)
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
        self.nn.opt_counter=manager.list([np.zeros([self.process])])  
        self.opt_counter_=manager.list()
        self._epoch_counter=manager.list([0 for _ in range(self.process)])
        self.nn.ec=manager.list([0])
        self.ec=self.nn.ec[0]
        self._batch_counter=manager.list([0 for _ in range(self.process)])
        self.nn.bc=manager.list([0])
        self.bc=self.nn.bc[0]
        self.epoch_=Value('i',0)
        self.stop_flag=Value('b',False)
        self.save_flag=Value('b',False)
        self.file_list=manager.list([])
        return
    
    
    def init_online(self,manager):
        self.nn.train_loss_list=manager.list([])
        self.nn.train_acc_list=manager.list([])
        self.nn.counter=manager.list([])
        self.nn.exception_list=manager.list([])
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
    
    
    def opt_p(self,data,labels,p):
        try:
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
        if self.priority_flag==True and self.priority_p.value!=-1:
            while True:
                if self.stop_flag.value==True:
                    return None,None
                if p==self.priority_p.value:
                    break
                else:
                    continue
        if self.stop_func_():
            return None,None
        self.nn.opt[p].zero_grad(set_to_none=True)
        loss=loss.clone()
        loss.backward()
        self.nn.opt[p].step()
        return output,loss
    
    
    def opt(self,data,labels,p):
        output,loss=self.opt_p(data,labels,p)
        return output,loss
    
    
    def train7(self,train_loader,p,test_batch,lock):
        while True:
            for data_batch,labels_batch in train_loader:
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
                    opt_counter[p]=0
                    self.nn.opt_counter[0]=opt_counter
                output,batch_loss=self.opt(data_batch,labels_batch,p)
                if self.stop_flag.value==True:
                    return
                if self.priority_flag==True:
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')
                    opt_counter+=1
                if hasattr(self.nn,'attenuate'):
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter+=1
                    self.nn.opt_counter[0]=opt_counter
                self.nn.bc[0]=sum(self._batch_counter)+self.bc
                _batch_counter=self._batch_counter[p]
                _batch_counter+=1
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
                batches=np.sum(self.batch_counter)
                if batches>=len(train_loader):
                    batch_counter=np.frombuffer(self.batch_counter.get_obj(),dtype='i')
                    batch_counter*=0
                    if lock is not None:
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
                    if lock is not None:
                        lock.release()
                    if self.test_flag==True:
                        if hasattr(self.nn,'accuracy'):
                            self.test_loss.value,self.test_acc.value=self.test(self.test_dataset,test_batch)
                            self.test_loss_list.append(self.test_loss.value)
                            self.test_acc_list.append(self.test_acc.value)
                        else:
                            self.test_loss.value=self.test(self.test_dataset,test_batch)
                            self.test_loss_list.append(self.test_loss.value)
                    self.save_()
                    self.epoch_counter.value+=1
                    self.nn.ec[0]=sum(self._epoch_counter)+self.ec
                    _epoch_counter=self._epoch_counter[p]
                    _epoch_counter+=1
                    self._epoch_counter[p]=_epoch_counter
                    total_loss=np.frombuffer(self.total_loss.get_obj(),dtype='f')
                    total_loss*=0
                    if hasattr(self.nn,'accuracy'):
                        total_acc=np.frombuffer(self.total_acc.get_obj(),dtype='f')
                        total_acc*=0
                if self.epoch!=None and self.epoch_counter.value>=self.epoch:
                    return
    
    
    def train(self,p,lock=None,test_batch=None):
        if self.prefetch_factor!=None:
            if type(self.train_dataset)==list:
                train_loader=torch.utils.data.DataLoader(self.train_dataset[p],batch_size=self.batch,prefetch_factor=self.prefetch_factor,num_workers=self.num_workers)
            else:
                train_loader=torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch,shuffle=self.shuffle,prefetch_factor=self.prefetch_factor,num_workers=self.num_workers)
        else:
            if type(self.train_dataset)==list:
                train_loader=torch.utils.data.DataLoader(self.train_dataset[p],batch_size=self.batch)
            else:
                train_loader=torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch,shuffle=self.shuffle)
        self.train7(train_loader,p,test_batch,lock)
        return
    
    
    def train_online(self,p,lock=None):
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
                output,loss,param=self.opt(data[0],data[1],p,lock)
                self.param[7]=param
            except Exception as e:
                if lock!=None:
                    if lock.acquire(False):
                        lock.release()
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
    
    
    def test(self,test_dataset,batch):
        total_loss=0
        total_acc=0
        batches=0
        test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch)
        for data_batch,labels_batch in test_loader:
            batches+=1
            batch_loss,batch_acc=self.test_(data_batch,labels_batch)
            total_loss+=batch_loss
            if hasattr(self.nn,'accuracy'):
                total_acc+=batch_acc
        test_loss=total_loss.detach().numpy()/batches
        if hasattr(self.nn,'accuracy'):
            test_acc=total_acc.detach().numpy()/batches
        if hasattr(self.nn,'accuracy'):
            return test_loss,test_acc
        else:
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
    
    
    def save_(self):
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
        plt.plot(np.arange(self.total_epoch.value),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('train loss:{0:.6f}'.format(self.train_loss.value))
        if hasattr(self.nn,'accuracy'):
            plt.figure(2)
            plt.plot(np.arange(self.total_epoch.value),self.train_acc_list)
            plt.title('train acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            if self.acc_flag=='%':
                print('train acc:{0:.1f}'.format(self.train_acc.value*100))
            else:
                print('train acc:{0:.6f}'.format(self.train_acc.value)) 
        return
    
    
    def visualize_test(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch.value),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('test loss:{0:.6f}'.format(self.test_loss.value))
        if hasattr(self.nn,'accuracy'):
            plt.figure(2)
            plt.plot(np.arange(self.total_epoch.value),self.test_acc_list)
            plt.title('test acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            if self.acc_flag=='%':
                print('test acc:{0:.1f}'.format(self.test_acc.value*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc.value))  
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
        if hasattr(self.nn,'accuracy'):
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
    
    
    def save_param(self,path):
        parameter_file=open(path,'wb')
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
        self.nn.opt_counter=self.nn.opt_counter[0] 
        self.nn.ec=self.nn.ec[0]
        self.nn.bc=self.nn.bc[0]
        self._epoch_counter=list(self._epoch_counter)
        self._batch_counter=list(self._batch_counter)
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
    
	
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        self.nn.km=1
        self.ec=self.nn.ec
        self.bc=self.nn.bc
        self.nn.opt_counter=self.opt_counter_
        self.nn.opt_counter.append(self.nn.opt_counter)
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
