import tensorflow as tf
import Note.create.TF2 as TF2
from tensorflow.python.ops import state_ops
import tensorflow.keras.optimizers as optimizers
import Note.create.optimizer as optimizern
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class unnamed:
    def __init__():
        tf2=TF2.tf2()
        with tf.name_scope('data/shape0'):
           
            
        with tf.name_scope('parameter'):
            
        
        with tf.name_scope('hyperparameter'):
            self.batch=None
            self.epoch=0
            self.lr=None
            
            
        self.regulation=None
        self.optimizer=None  
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_flag=False
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
    
    
    
    def train(self,batch=None,epoch=None,lr=None,model_path=None,one=True,processor=None):
        with tf.name_scope('hyperparameter'):
            self.batch=batch
            self.lr=lr
            
            
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        if processor!=None:
            self.processor=processor
        with tf.name_scope('processor_allocation'):
            
        
        with tf.device(train_processor):
            with tf.name_scope('variable'):
                
                
            with tf.name_scope('optimizer'):
                
                
            if self.total_epoch==0:
                epoch=epoch+1
            t1=time.time()
            for i in range(epoch):
                if batch!=None:
                    batches=int((self.shape0-self.shape0%batch)/batch)
                    tf2.batches=batches
                    total_loss=0
                    total_acc=0
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    with tf.name_scope('randomize_data'):
                        
                    
                    for j in range(batches):
                        tf2.index1=j*batch
                        tf2.index2=(j+1)*batch
                        with tf.name_scope('data_batch'):
                            
                        
                        with tf.GradientTape() as tape:
                            with tf.name_scope('forward_propagation/loss'):
                                
                        
                            if i==0 and self.total_epoch==0:
                                batch_loss=batch_loss.numpy()
                            else:
                                with tf.name_scope('apply_gradient'):
                                    
                                    
                        total_loss+=batch_loss
                        with tf.name_scope('accuracy'):
                     
                        
                        batch_acc=batch_acc.numpy()
                        total_acc+=batch_acc
                    if self.shape0%batch!=0:
                        batches+=1
                        tf2.batches+=1
                        tf2.index1=batches*batch
                        tf2.index2=batch-(self.shape0-batches*batch)
                        with tf.name_scope('data_batch'):
                            
                        
                        with tf.GradientTape() as tape:
                            with tf.name_scope('forward_propagation/loss'):
                                
                            
                            if i==0 and self.total_epoch==0:
                                batch_loss=batch_loss.numpy()
                            else:
                                with tf.name_scope('apply_gradient'):
                                    
                                    
                        total_loss+=batch_loss
                        with tf.name_scope('accuracy'):
                     
                        
                        batch_acc=batch_acc.numpy()
                        total_acc+=batch_acc
                    loss=total_loss/batches
                    train_acc=total_acc/batches
                    self.train_loss_list.append(loss.astype(np.float32))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float32)
                    self.train_acc_list.append(float(train_acc))
                    self.train_acc=train_acc
                    self.train_acc=self.train_acc.astype(np.float32)
                else:
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    with tf.name_scope('randomize_data'):
                        

                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            
                        
                        if i==0 and self.total_epoch==0:
                            loss=train_loss.numpy()
                        else:
                           with tf.name_scope('apply_gradient'):
                                
                                
                    self.train_loss_list.append(loss.astype(np.float32))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float32)
                    with tf.name_scope('accuracy'):
                     
                      
                    acc=train_acc.numpy()
                    self.train_acc_list.append(float(acc))
                    self.train_acc=acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if epoch%10!=0:
                    temp_epoch=epoch-epoch%10
                    temp_epoch=int(temp_epoch/10)
                else:
                    temp_epoch=epoch/10
                if temp_epoch==0:
                    temp_epoch=1
                if i%temp_epoch==0:
                    if self.total_epoch==0:
                        print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                    else:
                        print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                    if model_path!=None and i%epoch*2==0:
                        self.save(model_path,i,one)
            t2=time.time()
            _time=(t2-t1)-int(t2-t1)
            if self.time==0:
                self.total_time=_time
            else:
                self.total_time+=_time
            if _time<0.5:
                self.time=int(t2-t1)
            else:
                self.time=int(t2-t1)+1
            print()
            print('last loss:{0:.6f}'.format(self.train_loss))
            with tf.name_scope('print_accuracy'):
                    
            
            if self.total_epoch==0:
                self.total_epoch=epoch-1
                self.epoch=epoch-1
            else:
                self.total_epoch=self.total_epoch+epoch
                self.epoch=epoch
            print('time:{0}s'.format(self.time))
            return
    
    
    def test(self,test_data,test_labels,batch=None):
        self.test_flag=True
        batch_temp=self.batch
        self.batch=batch
        if batch!=None:
            total_loss=0
            total_acc=0
            batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
            self.batches=batches
            for j in range(batches):
                self.index1=j*batch
                self.index2=(j+1)*batch
                with tf.name_scope('data_batch'):
                    
                    
                with tf.name_scope('loss'):
                
                    
                total_loss+=batch_test_loss.numpy()
                with tf.name_scope('accuracy'):
                    
                
                total_acc+=batch_acc.numpy()
            if test_data.shape[0]%batch!=0:
                batches+=1
                self.batches+=1
                with tf.name_scope('data_batch'):
                    
                    
                with tf.name_scope('loss'):
                    
                
                total_loss+=batch_loss.numpy()
                with tf.name_scope('accuracy'):
                    
                
                total_acc+=batch_acc.numpy()
            test_loss=total_loss/batches
            test_acc=total_acc/batches
            self.test_loss=test_loss
            self.test_acc=test_acc
            self.test_loss=self.test_loss.astype(np.float32)
            self.test_acc=self.test_acc.astype(np.float32)
        else:
            with tf.name_scope('loss'):
                
                
            with tf.name_scope('accuracy'):
                
                
            self.test_loss=test_loss.numpy().astype(np.float32)
            self.test_acc=test_acc.numpy().astype(np.float32)
        print('test loss:{0:.6f}'.format(self.test_loss))
        with tf.name_scope('print_accuracy'):
            
        self.batch=batch_temp    
        return
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.epoch))
        print()
        print('optimizer:{0}'.format(self.optimizer))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0:.6f}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
                
                
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        with tf.name_scope('print_accuracy'):
            
        
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
        plt.plot(np.arange(self.epoch+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.train_acc_list)
        plt.title('train acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('train loss:{0:.6f}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
                
                
        return
    
        
    def comparison(self):
        print()
        print('train loss:{0}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
                
            
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            with tf.name_scope('print_accuracy'):
                
                
        return
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        with tf.name_scope('save_parameter'):  
            

        with tf.name_scope('save_hyperparameter'):
            pickle.dump(self.batch,output_file)
            pickle.dump(self.epoch,output_file)
            pickle.dump(self.lr,output_file)
            
            
        pickle.dump(self.regulation,output_file)    
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_acc,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_acc,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_acc_list,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.time,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        input_file=open(model_path,'rb')
        with tf.name_scope('restore_parameter'):
            
            
        with tf.name_scope('restore_hyperparameter'):
            self.batch=pickle.load(input_file)
            self.epoch=pickle.load(input_file)
            self.lr=pickle.load(input_file)
            
            
        self.regulation=pickle.load(input_file)    
        self.optimizer=pickle.load(input_file)
        self.shape0=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_acc=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_acc=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_acc_list=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        input_file.close()
        return
