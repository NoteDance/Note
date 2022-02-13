import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        self.param=nn.param
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
        self.acc_flag1=nn.acc_flag1
        self.acc_flag2=nn.acc_flag2
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
        return
    
    
    def init(self,param=None):
        if param!=None:
            self.param=param
        self.flag=None
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
    
    
    def apply_gradient(tape,opt,loss,parameter):
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
    
    
    def _train(self,batch=None,test_batch=None,data_batch=None,labels_batch=None):
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
                    output=self.nn.fp(data_batch)
                    batch_loss=self.nn.loss(output,labels_batch)
                if self.optf!=True:
                    self.apply_gradient(tape,self.opt,batch_loss,self.param)
                else:
                    gradient=tape.gradient(batch_loss,self.param)
                    self.sopt(gradient,self.param)
                if self.total_epoch==1:
                    batch_loss=batch_loss.numpy()
                total_loss+=batch_loss
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                if self.bflag==True:
                    self.nn.batchcount=j
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
                    output=self.nn.fp(data_batch)
                    batch_loss=self.nn.loss(output,labels_batch)
                if self.optf!=True:
                    self.apply_gradient(tape,self.opt,batch_loss,self.param)
                else:
                    gradient=tape.gradient(batch_loss,self.param)
                    self.sopt(gradient,self.param)
                if self.total_epoch==1:
                    batch_loss=batch_loss.numpy()
                total_loss+=batch_loss
                if self.acc_flag1==1:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                if self.bflag==True:
                    self.nn.batchcount+=1
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
        elif self.ol==None:
            with tf.GradientTape() as tape:
                output=self.nn.fp(self.train_data)
                train_loss=self.nn.loss(output,self.train_labels)
            if self.optf!=True:
                self.apply_gradient(tape,self.opt,train_loss,self.param)
            else:
                gradient=tape.gradient(train_loss,self.param)
                self.sopt(gradient,self.param)
            if self.total_epoch==1:
                loss=train_loss.numpy()
            self.train_loss_list.append(loss.astype(np.float32))
            self.train_loss=loss
            self.train_loss=self.train_loss.astype(np.float32)
            if self.acc_flag1==1:
                acc=self.nn.accuracy(output,self.train_labels)
                acc=train_acc.numpy()
                self.train_acc_list.append(acc.astype(np.float32))
                self.train_acc=acc
                self.train_acc=self.train_acc.astype(np.float32)
            if self.test==True:
                self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                self.test_loss_list.append(self.test_loss)
                if self.acc_flag1==1:
                    self.test_acc_list.append(self.test_acc)
        else:
            data=self.ol()
            if data=='end':
                return
            self.total_epoch+=1
            with tf.GradientTape() as tape:
                output=self.nn.fp(data[0])
                train_loss=self.nn.loss(output,data[1])

            if self.optf!=True:
                self.apply_gradient(tape,self.opt,train_loss,self.param)
            else:
                gradient=tape.gradient(train_loss,self.param)
                self.sopt(gradient,self.param)
            if self.total_epoch==1:
                loss=train_loss.numpy()
            self.nn.train_loss=loss.astype(np.float32)
            if self.eflag==True:
                self.nn.epochcount+=1
        return
        
    
    def train(self,batch=None,epoch=None,test_batch=None,nn_path=None,one=True):
        self.batch=batch
        self.epoch=0
        self.optf=self.nn.optf
        if self.flag==None:
            self.flag=True
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
                    self.nn.epochcount+=1
                self._train(batch,test_batch,data_batch,labels_batch)
                self.epoch+=1
                self.total_epoch+=1
                if epoch%10!=0:
                    d=epoch-epoch%9
                    d=int(d/9)
                else:
                    d=epoch/10
                if d==0:
                    d=1
                e=d*2
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
                self.time+=(t2-t1)
                if self.end()==True:
                    break
        elif self.ol==None:
            i=0
            while True:
                t1=time.time()
                i+=1
                self._train(batch,test_batch)
                self.epoch+=1
                self.total_epoch+=1
                if epoch%10!=0:
                    d=epoch-epoch%9
                    d=int(d/9)
                else:
                    d=epoch/10
                if d==0:
                    d=1
                e=d*2
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
                    self.nn.epochcount+=1
                t2=time.time()
                self.time+=(t2-t1)
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
        self.time=self.time-int(self.time)
        if self.time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
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
        return
    
    
    def test(self,test_data,test_labels,batch=None):
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
                output=self.nn.fp(data_batch)
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
                output=self.nn.fp(data_batch)
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
            output=self.nn.fp(test_data)
            test_loss=self.nn.loss(output,test_labels)
            if self.acc_flag1==1:
                test_acc=self.nn.accuracy(output,test_labels)
                test_loss=test_loss.numpy().astype(np.float32)
                test_acc=test_acc.numpy().astype(np.float32)
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
        pickle.dump(self.param,parameter_file)
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
        pickle.dump(self.param,parameter_file)
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
        pickle.dump(self.shape0,output_file)
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
        self.param=pickle.load(parameter_file)
        self.nn=pickle.load(input_file)
        self.nn.param=self.param
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
        self.shape0=pickle.load(input_file)
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
