import tensorflow as tf
import Note.create.creat as c
from tensorflow.python.ops import state_ops
import tensorflow.keras.optimizers as optimizers
import Note.create.optimizer as optimizern
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class unnamed:
    def __init__():
        with tf.name_scope('data/shape0'):
           
            
        with tf.name_scope('parameter'):
            
        
        with tf.name_scope('hyperparameter'):
            self.batch=None
            self.epoch=0
            self.lr=None
            
            
        with tf.name_scope('regulation'):   
            self.regulation=None   
        with tf.name_scope('optimizer'):
            self.opt=None
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.buffer_size=None
        self.ooo=False
        self.total_epoch=0
        self.time=0
        self.total_time=0
        self.processor='GPU:0'
        
    
    def weight_init(self,shape,mean,stddev):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
            
    
    def bias_init(self,shape,mean,stddev):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
    
    
    def structure():
        self.test_flag=False
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        self.test_loss_list.clear()
        self.test_acc_list.clear()
        self.dtype=dtype
        with tf.name_scope('hyperparameter'):
            self.epoch=0
            
            
        self.total_epoch=0    
        self.time=0
        self.total_time=0
        with tf.name_scope('parameter_initialization'):
            
           
            
    @tf.function       
    def forward_propagation():
        with tf.name_scope('processor_allocation'):
               
               
        with tf.name_scope('forward_propagation'):
    
            
            
    def set_up(self,optimizer=None,optimizern=None,lr=None,l2=None,dropout=None):
        with tf.name_scope('hyperparameter'):
            if optimizer!=None or optimizern!=None:
                self.optimizer=optimizer
                self.optimizern=optimizern
                if optimizer!=None:
                    self.lr=optimizer.lr
                else:
                    self.lr=optimizern.lr
            if self.optimizer!=None and lr!=None:
                self.optimizer.lr=lr
                self.lr=lr
            elif lr!=None:
                self.optimizern.lr=lr
                self.lr=lr
            if l2!=None:
                self.l2=l2
            if dropout!=None:
                self.dropout=dropout
            return
        
    
    def train(self,batch=None,epoch=None,test=False,test_batch=None,model_path=None,one=True,buffer_size=None,processor=None):
        with tf.name_scope('hyperparameter'):
            self.batch=batch
            self.epoch=0
            
        
        batches=int((self.shape0-self.shape0%batch)/batch)
        if self.shape0%batch!=0:
            batches+=1
        if buffer_size!=None:
            self.buffer_size=buffer_size
        elif self.buffer_size!=None:
            pass
        else:
            self.buffer_size=self.shape0
        self.time=0
        self.test_flag=test
        if processor!=None:
            self.processor=processor
        with tf.name_scope('parameter'):
            
            
        if self.total_epoch==0:
            epoch=epoch+1
        for i in range(epoch):
            t1=time.time()
            if batch!=None:
                total_loss=0
                total_acc=0
                train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).shuffle(self.buffer_size).batch(batch)
                for data_batch,labels_batch in train_ds:
                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            
                    
                    if i==0 and self.total_epoch==0:
                        batch_loss=batch_loss.numpy()
                    else:
                        with tf.name_scope('apply_gradient'):
                            if self.optimizer!=None:
                                c.apply_gradient(tape,self.optimizer,batch_loss,parameter)
                            else:
                                gradient=tape.gradient(batch_loss,parameter)
                                self.optimizern(gradient,parameter)
                    total_loss+=batch_loss
                    with tf.name_scope('accuracy'):
                 
                    
                    batch_acc=batch_acc.numpy()
                    total_acc+=batch_acc
                loss=total_loss/batches
                train_acc=total_acc/batches
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                self.train_acc_list.append(train_acc.astype(np.float32))
                self.train_acc=train_acc
                self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_acc_list.append(self.test_acc)
            else:
                with tf.GradientTape() as tape:
                    with tf.name_scope('forward_propagation/loss'):
                        
                    
                if i==0 and self.total_epoch==0:
                    loss=train_loss.numpy()
                else:
                   with tf.name_scope('apply_gradient'):
                       if self.optimizer!=None:
                           c.apply_gradient(tape,self.optimizer,batch_loss,parameter)
                       else:
                           gradient=tape.gradient(batch_loss,parameter)
                           self.optimizern(gradient,parameter)
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                with tf.name_scope('accuracy'):
                 
                  
                acc=train_acc.numpy()
                self.train_acc_list.append(acc.astype(np.float32))
                self.train_acc=acc
                self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                        self.test_loss_list.append(self.test_loss)
                        self.test_acc_list.append(self.test_acc)
            self.epoch+=1
            self.total_epoch+=1
            if epoch%10!=0:
                d=epoch-epoch%10
                d=int(d/10)
            else:
                d=epoch/10
            if d==0:
                d=1
            if i%d==0:
                if self.total_epoch==0:
                    print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                else:
                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                if model_path!=None and i%epoch*2==0:
                    self.save(model_path,i,one)
            t2=time.time()
            self.time+=(t2-t1)
        if model_path!=None:
            self.save(model_path)
        self.time=self.time-int(self.time)
        if self.time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
            print('accuracy:{0:.1f}'.format(self.train_acc*100))
            print('accuracy:{0:.6f}'.format(self.train_acc))   
        print('time:{0}s'.format(self.time))
        return
    
    
    def test(self,test_data,test_labels,batch=None,buffer_size=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
            if test_data.shape[0]%batch!=0:
                batches+=1
            if buffer_size!=None:
                buffer_size=buffer_size
            else:
                buffer_size=len(test_data)
            test_ds=tf.data.Dataset.from_tensor_slices((test_data,test_labels)).batch(batch)
            for data_batch,labels_batch in test_ds:
                with tf.name_scope('loss'):
                    
                    
                total_loss+=batch_loss.numpy()
                with tf.name_scope('accuracy'):
                    
                    
                total_acc+=batch_acc.numpy()
            test_loss=total_loss/batches
            test_acc=total_acc/batches
            test_loss=test_loss
            test_acc=test_acc
            test_loss=test_loss.astype(np.float32)
            test_acc=test_acc.astype(np.float32)
        else:
            with tf.name_scope('loss'):
                
                
            with tf.name_scope('accuracy'):
                
                
            test_loss=test_loss.numpy().astype(np.float32)
            test_acc=test_acc.numpy().astype(np.float32)
        return test_loss,test_acc
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.total_epoch))
        if self.regulation!=None:
            print()
            print('regulation:{0}'.format(self.regulation))
        if self.optimizer!=None:
            print()
            print('optimizer:{0}'.format(self.opt))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.total_time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0:.6f}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
            print('train acc:{0:.1f}'.format(self.train_acc*100))
            print('train acc:{0:.6f}'.format(self.train_acc))       
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        with tf.name_scope('print_accuracy'):
            print('test acc:{0:.1f}'.format(self.test_acc*100))
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
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.train_acc_list)
        plt.title('train acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('train loss:{0:.6f}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
            print('train acc:{0:.1f}'.format(self.train_acc*100))
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
        with tf.name_scope('print_accuracy'):
            print('test acc:{0:.1f}'.format(self.test_acc*100))
            print('test acc:{0:.6f}'.format(self.test_acc))  
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
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.train_acc_list,'b-',label='train acc')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch),self.test_acc_list,'r-',label='test acc')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        print('train loss:{0}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
            print('train acc:{0:.1f}'.format(self.train_acc*100))
            print('train acc:{0:.6f}'.format(self.train_acc))     
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            with tf.name_scope('print_accuracy'):
                print('test acc:{0:.1f}'.format(self.test_acc*100))
                print('test acc:{0:.6f}'.format(self.test_acc)) 
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        parameter=pickle.dump(,parameter_file)
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
        with tf.name_scope('save_parameter'):  
            

        with tf.name_scope('save_hyperparameter'):
            pickle.dump(self.batch,output_file)
            pickle.dump(self.lr,output_file)
            
            
        with tf.name_scope('save_regulation'):
            pickle.dump(self.regulation,output_file)   
        with tf.name_scope('save_optimizer'):
            pickle.dump(self.opt,output_file)
            if self.optimizer!=None:
                pickle.dump(self.optimizer,output_file)
            else:
                pickle.dump(self.optimizern,output_file)
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
        pickle.dump(self.buffer_size,output_file) 
        pickle.dump(self.ooo,output_file)    
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        parameter_file.close()
        return
    

    def restore(self,s_path,p_path):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        with tf.name_scope('restore_parameter'):
            
            
        with tf.name_scope('restore_hyperparameter'):
            self.batch=pickle.load(input_file)
            self.lr=pickle.load(input_file)
            
            
        with tf.name_scope('restore_regulation'):
            self.regulation=pickle.load(input_file)
        with tf.name_scope('restore_optimizer'):
            self.opt=pickle.load(input_file)
            if self.optimizer!=None:
                self.optimizer=pickle.load(input_file)
            else:
                self.optimizern=pickle.load(input_file)
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
        self.buffer_size=pickle.load(input_file)
        self.ooo=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        input_file.close()
        parameter_file.close()
        return
