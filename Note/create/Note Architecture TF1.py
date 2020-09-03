import tensorflow as tf
import Note.create.create as c
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class unnamed:
    def __init__():
        self.graph=tf.Graph()
        tf1=c.tf1()
        with tf.name_scope('data/shape0'):
            
            
        with self.graph.as_default():
            with tf.name_scope('placeholder'):
                
        
        with tf.name_scope('parameter'):
            
            
        with tf.name_scope('hyperparameter'):    
            self.batch=None
            self.epoch=0
            self.lr=None
            
        with tf.name_scope('regulation'):   
            self.regulation=None   
        with tf.name_scope('optimizer'):
            self.optimizer=None
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.total_epoch=0
        self.time=0
        self.total_time=0
        self.processor='/gpu:0'
        
    
    def weight_init(self,shape,mean,stddev,name=None):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
            
    
    def bias_init(self,shape,mean,stddev,name=None):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
    
    
    def structure():
        with self.graph.as_default():
            self.continue_train=False
            self.flag=None
            self.end_flag=False
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
                
                
                
    def forward_propagation():
        with self.graph.as_default():
           with tf.name_scope('processor_allocation'):
               
               
           with tf.name_scope('forward_propagation'):
            
               
               
    def train(self,batch=None,epoch=None,lr=None,test=False,test_batch=None,train_summary_path=None,model_path=None,one=True,continue_train=False,processor=None):
        with self.graph.as_default():
            with tf.name_scope('hyperparameter'):
                self.batch=batch
                self.lr=lr
                
                
            self.test_flag=test   
            if continue_train!=True:
                if self.continue_train==True:
                    continue_train=True
                else:
                    self.train_loss_list.clear()
                    self.train_acc_list.clear()
                    self.test_loss_list.clear()
                    self.test_acc_list.clear()
            if self.continue_train==False and continue_train==True:
                self.train_loss_list.clear()
                self.train_acc_list.clear()
                self.test_loss_list.clear()
                self.test_acc_list.clear()
                self.continue_train=True
            if processor!=None:
                self.processor=processor
            if continue_train==True and self.end_flag==True:
                self.end_flag=False
                with tf.name_scope('parameter_convert_into_tensor):
                
                
            if continue_train==True and self.flag==1:
                self.flag=0
                with tf.name_scope('parameter_convert_into_tensor):
                
                    
            with tf.name_scope('forward_propagation'):
            
            
            with tf.name_scope('train_loss'):
					
                
                train_loss_scalar=tf.summary.scalar('train_loss',train_loss)
            with tf.name_scope('optimizer'): 
                
                                        
            with tf.name_scope('train_accuracy'):
                
                
                train_acc_scalar=tf.summary.scalar('train_accuracy',train_acc)
            if train_summary_path!=None:
                train_merging=tf.summary.merge([train_loss_scalar,train_acc_scalar])
                train_writer=tf.summary.FileWriter(train_summary_path)
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            sess=tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            self.sess=sess
            if self.total_epoch==0:
                epoch=epoch+1
            t1=time.time()
            for i in range(epoch):
                if batch!=None:
                    batches=int((self.shape0-self.shape0%batch)/batch)
                    tf1.batches=batches
                    total_loss=0
                    total_acc=0
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    with tf.name_scope('randomize_data'):
                    
                        
                    for j in range(batches):
                        tf1.index1=j*batch
                        tf1.index2=(j+1)*batch
                        with tf.name_scope('data_batch/feed_dict'):
                        
                        
                        if i==0 and self.total_epoch==0:
                            batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                        else:
                            batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                        total_loss+=batch_loss
                        batch_acc=sess.run(train_acc,feed_dict=feed_dict)
                        total_acc+=batch_acc
                    if self.shape0%batch!=0:
                        batches+=1
                        tf1.batches+=1
                        tf1.index1=batches*batch
                        tf1.index2=batch-(self.shape0-batches*batch)
                        with tf.name_scope('data_batch/feed_dict'):
                            
                        
                        if i==0 and self.total_epoch==0:
                            batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                        else:
                            batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                        total_loss+=batch_loss
                        batch_acc=sess.run(train_acc,feed_dict=feed_dict)
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
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    with tf.name_scope('randomize_data/feed_dict'):
                        

                    if i==0 and self.total_epoch==0:
                        loss=sess.run(train_loss,feed_dict=feed_dict)
                    else:
                        loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                    self.train_loss_list.append(loss.astype(np.float32))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float32)
                    acc=sess.run(train_acc,feed_dict=feed_dict)
                    self.train_acc_list.append(acc.astype(np.float32))
                    self.train_acc=acc
                    self.train_acc=self.train_acc.astype(np.float32)
                    if test==True:
                        with tf.name_scope('test'):
                            self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                            self.test_loss_list.append(self.test_loss)
                            self.test_acc_list.append(self.test_acc)
                if epoch%10!=0:
                    temp=epoch-epoch%10
                    temp=int(temp/10)
                else:
                    temp=epoch/10
                if temp==0:
                    temp=1
                if i%temp==0:
                    if continue_train==True:
                        print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                    else:
                        print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                    if model_path!=None and i%epoch*2==0:
                        self.save(model_path,i,one)
                    if train_summary_path!=None:
                        train_summary=sess.run(train_merging,feed_dict=feed_dict)
                        train_writer.add_summary(train_summary,i)
            t2=time.time()
            _time=(t2-t1)-int(t2-t1)
            if _time<0.5:
                self.time=int(t2-t1)
            else:
                self.time=int(t2-t1)+1
            self.total_time+=self.time
            print()
            print('last loss:{0:.6f}'.format(self.train_loss))
            with tf.name_scope('print_accuracy'):
                print('accuracy:{0:.1f}'.format(self.train_acc*100))
                print('accuracy:{0:.6f}'.format(self.train_acc))  
            if train_summary_path!=None:
                train_writer.close()
            if continue_train==True:
                with tf.name_scope('parameter_convert_into_numpy):
                    
                
                sess.run(tf.global_variables_initializer())
            if continue_train==True:
                if self.total_epoch==0:
                    self.total_epoch=epoch-1
                    self.epoch=epoch-1
                else:
                    self.total_epoch=self.total_epoch+epoch
                    self.epoch=epoch
            if continue_train!=True:
                self.epoch=epoch-1
            print('time:{0}s'.format(self.time))
            return
    
    
    def end(self):
        with self.graph.as_default():
            self.end_flag=True
            self.continue_train=False
            with tf.name_scope('parameter_convert_into_numpy'):
                
                
            self.sess.close()
            return
    
    
    def test(self,test_data,test_labels,batch=None):
        with self.graph.as_default():
            with tf.name_scope('placeholder'):
                
                
            with tf.name_scope('loss'):
                
                
            with tf.name_scope('accuracy'):
                
                
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            sess=tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            if batch!=None:
                total_loss=0
                total_acc=0
                batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
                tf1.batches=batches
                for j in range(batches):
                    tf1.index1=j*batch
                    tf1.index2=(j+1)*batch
                    with tf.name_scope('data_batch/feed_dict'):
                        
                        
                    batch_loss=sess.run(test_loss,feed_dict=feed_dict)
                    total_loss+=batch_loss
                    batch_acc=sess.run(test_acc,feed_dict=feed_dict)
                    total_acc+=batch_acc
                if test_data.shape[0]%batch!=0:
                    batches+=1
                    tf1.batches+=1
                    tf1.index1=batches*batch
                    tf1.index2=batch-(self.shape0-batches*batch)
                    with tf.name_scope('data_batch/feed_dict'):
                        
                        
                    batch_loss=sess.run(test_loss,feed_dict=feed_dict)
                    total_loss+=batch_loss
                    batch_acc=sess.run(test_acc,feed_dict=feed_dict)
                    total_acc+=batch_acc
                test_loss=total_loss/batches
                test_acc=total_acc/batches
                test_loss=test_loss
                test_acc=test_acc
                test_loss=test_loss.astype(np.float32)
                test_acc=test_acc.astype(np.float32)
            else:
                with tf.name_scope('feed_dict'):
                    
                    
                test_loss=sess.run(test_loss,feed_dict=feed_dict)
                test_acc=sess.run(test_acc,feed_dict=feed_dict)
                test_loss=test_loss.astype(np.float32)
                test_acc=test_acc.astype(np.float32)
            sess.close()
            return test_loss,test_acc
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.epoch))
        if self.regulation!=None:
            print()
            print('regulation:{0}'.format(self.regulation))
        if self.optimizer!=None:
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
            print('train acc:{0:.1f}'.format(self.train_acc*100))
            print('train acc:{0:.6f}'.format(self.train_acc))      
        return
    
    
    def test_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.test_acc_list)
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
        plt.plot(np.arange(self.epoch+1),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(self.epoch+1),self.test_loss_list,'r-',label='test loss')
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.train_acc_list,'b-',label='train acc')
        if self.test_flag==True:
            plt.plot(np.arange(self.epoch+1),self.test_acc_list,'r-',label='test acc')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        print('train loss:{0:.6f}'.format(self.train_loss))
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
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        with tf.name_scope('save_parameter'):
            
          
        with tf.name_scope('save_data_msg'):
            pickle.dump(tf1.dtype)
            pickle.dump(tf1.shape)           
        with tf.name_scope('save_hyperparameter'):
            pickle.dump(self.batch,output_file)
            pickle.dump(self.epoch,output_file)
            pickle.dump(self.lr,output_file)
            
        with tf.name_scope('save_regulation'):    
            pickle.dump(self.regulation,output_file)     
        with tf.name_scope('save_optimizer'):
            pickle.dump(self.optimizer,output_file)
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
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.time,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        input_file=open(model_path,'rb')
        tf1.accumulator=0
        tf1.test_accumulator=0
        with tf.name_scope('restore_parameter'):
            
            
        with tf.name_scope('restore_data_msg'):    
            tf1.dtype=pickle.load(input_file)
            tf1.shape=pickle.load(input_file)
        self.graph=tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('placeholder'):
                
                
        with tf.name_scope('restore_hyperparameter'):
            self.batch=pickle.load(input_file)
            self.epoch=pickle.load(input_file)
            self.lr=pickle.load(input_file)
            
        with tf.name_scope('restore_regulation'):    
            self.regulation=pickle.load(input_file)
        with tf.name_scope('restore_optimizer'):
            self.optimizer=pickle.load(input_file)    
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
        self.total_epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        self.flag=1
        input_file.close()
        return
