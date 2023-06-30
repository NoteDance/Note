import tensorflow as tf
import numpy as np
from multiprocessing import Array
import numpy.ctypeslib as npc


def test(nn,test_data,test_labels,platform,batch=None,loss=None,acc_flag='%'):
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
            try:
                try:
                    output=nn.fp(data_batch)
                except Exception:
                    output=nn(data_batch)
            except Exception as e:
                raise e
            if loss==None:
                batch_loss=nn.loss(output,labels_batch)
            else:
                batch_loss=loss(labels_batch,output)
            total_loss+=batch_loss
            try:
                batch_acc=nn.accuracy(output,labels_batch)
                total_acc+=batch_acc
            except Exception as e:
                try:
                    if nn.accuracy!=None:
                        raise e
                except Exception:     
                    pass
        if shape0%batch!=0:
            batches+=1
            index1=batches*batch
            index2=batch-(shape0-batches*batch)
            try:
                try:
                    if type(test_data)==list:
                        for i in range(len(test_data)):
                            data_batch[i]=platform.concat([test_data[i][index1:],test_data[i][:index2]],0)
                    else:
                        data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)
                    if type(test_labels)==list:
                        for i in range(len(test_labels)):
                            labels_batch[i]=platform.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                    else:
                        labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)
                except Exception:
                    if type(test_data)==list:
                        for i in range(len(test_data)):
                            data_batch[i]=platform.concat([test_data[i][index1:],test_data[i][:index2]],0)
                    else:
                        data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)
                    if type(test_labels)==list:
                        for i in range(len(test_labels)):
                            labels_batch[i]=platform.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                    else:
                        labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)
            except Exception as e:
                raise e
            try:
                try:
                    output=nn.fp(data_batch)
                except Exception:
                    output=nn(data_batch)
            except Exception as e:
                raise e
            if loss==None:
                batch_loss=nn.loss(output,labels_batch)
            else:
                batch_loss=loss(labels_batch,output)
            total_loss+=batch_loss
            try:
                batch_acc=nn.accuracy(output,labels_batch)
                total_acc+=batch_acc
            except Exception as e:
                try:
                  if nn.accuracy!=None:
                      raise e
                except Exception: 
                    pass
        test_loss=total_loss.numpy()/batches
        test_loss=test_loss.astype(np.float32)
        try:
            if nn.accuracy!=None:
                test_acc=total_acc.numpy()/batches
                test_acc=test_acc.astype(np.float32)
        except Exception:
            pass
    else:
        try:
            try:
                output=nn.fp(test_data)
            except Exception:
                output=nn(test_data)
        except Exception as e:
            raise e
        if loss==None:
            test_loss=nn.loss(output,test_labels)
        else:
            test_loss=loss(test_labels,output)
        test_loss=test_loss.numpy().astype(np.float32)
        try:
            test_acc=nn.accuracy(output,test_labels)
            test_acc=test_acc.numpy().astype(np.float32)
        except Exception as e:
            try:
                if nn.accuracy!=None:
                    raise e
            except Exception: 
                pass
    print('test loss:{0:.6f}'.format(test_loss))
    try:
        if nn.accuracy!=None:
            if acc_flag=='%':
                print('accuracy:{0:.1f}'.format(test_acc*100))
            else:
                print('accuracy:{0:.6f}'.format(test_acc))
            if acc_flag=='%':
                return test_loss,test_acc*100
            else:
                return test_loss,test_acc
    except Exception:
        return test_loss


class parallel_test:
    def __init__(self,nn,test_data=None,test_labels=None,process=None,batch=None):
        self.nn=nn
        if type(self.nn.param[0])!=list:
            self.test_data=test_data.astype(self.nn.param[0].dtype.name)
            self.test_labels=test_labels.astype(self.nn.param[0].dtype.name)
        else:
            self.test_data=test_data.astype(self.nn.param[0][0].dtype.name)
            self.test_labels=test_labels.astype(self.nn.param[0][0].dtype.name)
        self.process=process
        self.batch=batch
        if type(self.nn.param[0])!=list:
            self.loss=Array('f',np.zeros([process],dtype=self.nn.param[0].dtype.name))
        else:
            self.loss=Array('f',np.zeros([process],dtype=self.nn.param[0][0].dtype.name))
        try:
            if self.nn.accuracy!=None:
                if type(self.nn.param[0])!=list:
                    self.acc=Array('f',np.zeros([process],dtype=self.nn.param[0].dtype.name))
                else:
                    self.acc=Array('f',np.zeros([process],dtype=self.nn.param[0][0].dtype.name))
        except Exception:
            pass
        self.process_num=np.arange(process)
        self.process_num=list(self.process_num)
    
    
    def segment_data(self):
        if len(self.test_data)!=self.process:
            data=None
            labels=None
            segments=int((len(self.test_data)-len(self.test_data)%self.process)/self.process)
            for i in range(self.process):
                index1=i*segments
                index2=(i+1)*segments
                if i==0:
                    data=np.expand_dims(self.test_data[index1:index2],axis=0)
                    labels=np.expand_dims(self.test_labels[index1:index2],axis=0)
                else:
                    data=np.concatenate((data,np.expand_dims(self.test_data[index1:index2],axis=0)))
                    labels=np.concatenate((labels,np.expand_dims(self.test_labels[index1:index2],axis=0)))
            if len(data)%self.process!=0:
                segments+=1
                index1=segments*self.process
                index2=self.process_thread-(len(self.train_data)-segments*self.process)
                data=np.concatenate((data,np.expand_dims(self.test_data[index1:index2],axis=0)))
                labels=np.concatenate((labels,np.expand_dims(self.test_labels[index1:index2],axis=0)))
            self.test_data=data
            self.test_labels=labels
        return
    
    
    def test(self):
        p=self.process_num.pop(0)
        if type(self.test_data)==list:
            train_ds=self.test_data[p]
        else:
            train_ds=tf.data.Dataset.from_tensor_slices((self.test_data[p],self.test_labels[p])).batch(self.batch)
        for data_batch,labels_batch in train_ds:
            try:
                try:
                    output=self.nn.fp(data_batch)
                    batch_loss=self.nn.loss(output,labels_batch)
                except Exception:
                    output=self.nn.fp(data_batch,p)
                    batch_loss=self.nn.loss(output,labels_batch)
            except Exception as e:
                raise e
            try:
                batch_acc=self.nn.accuracy(output,labels_batch)
            except Exception as e:
                try:
                    if self.nn.accuracy!=None:
                        raise e
                except Exception:
                    pass
            try:
                if self.nn.accuracy!=None:
                    self.loss[p]+=batch_loss
                    self.acc[p]+=batch_acc
            except Exception:
                self.loss[p]+=batch_loss
        return
    
    
    def loss_acc(self):
        if type(self.test_data)==list:
            shape=len(self.test_data[0])*self.batch
        else:
            shape=len(self.test_data[0])
        batches=int((shape-shape%self.batch)/self.batch)
        if shape%self.batch!=0:
            batches+=1
        try:
            if self.nn.accuracy!=None:
                return np.mean(npc.as_array(self.loss.get_obj())/batches),np.mean(npc.as_array(self.acc.get_obj())/batches)
        except Exception:
            return np.mean(npc.as_array(self.loss.get_obj())/batches)
