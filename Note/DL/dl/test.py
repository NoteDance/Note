import tensorflow as tf
import numpy as np
from multiprocessing import Array
import numpy.ctypeslib as npc


@tf.function(jit_compile=True)
def test_tf(nn,data,labels):
    try:
        try:
            output=nn.fp(data)
            loss=nn.loss(output,labels)
        except Exception:
            output,loss=nn.fp(data,labels)
    except Exception as e:
        raise e
    try:
        if hasattr(nn,'accuracy'):
            acc=nn.accuracy(output,labels)
        else:
            acc=None
    except Exception as e:
        raise e
    return loss,acc


def test_pytorch(nn,data,labels):
    try:
        try:
            output=nn.fp(data)
            loss=nn.loss(output,labels)
        except Exception:
            output,loss=nn.fp(data,labels)
    except Exception as e:
        raise e
    try:
        if hasattr(nn,'accuracy'):
            acc=nn.accuracy(output,labels)
        else:
          acc=None  
    except Exception as e:
        raise e
    return loss,acc


def test(nn,test_data,test_labels,platform,batch=None,loss=None,acc_flag='%'):
    if batch!=None:
        total_loss=0
        total_acc=0
        batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
        shape0=test_data.shape[0]
        for j in range(batches):
            index1=j*batch
            index2=(j+1)*batch
            data_batch=test_data[index1:index2]
            labels_batch=test_labels[index1:index2]
            if hasattr(platform,'DType'):
                batch_loss,batch_acc=test_tf(data_batch,labels_batch)
            else:
                batch_loss,batch_acc=test_pytorch(data_batch,labels_batch)
            total_loss+=batch_loss
            if hasattr(nn,'accuracy'):
                total_acc+=batch_acc
        if shape0%batch!=0:
            batches+=1
            index1=batches*batch
            index2=batch-(shape0-batches*batch)
            try:
                try:
                    data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)
                    labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)
                except Exception:
                    data_batch=np.concatenate([test_data[index1:],test_data[:index2]],0)
                    labels_batch=np.concatenate([test_labels[index1:],test_labels[:index2]],0)
            except Exception as e:
                raise e
            if hasattr(platform,'DType'):
                batch_loss,batch_acc=test_tf(data_batch,labels_batch)
            else:
                batch_loss,batch_acc=test_pytorch(data_batch,labels_batch)
            total_loss+=batch_loss
            if hasattr(nn,'accuracy'):
                total_acc+=batch_acc
        test_loss=total_loss.numpy()/batches
        test_loss=test_loss.astype(np.float32)
        if hasattr(nn,'accuracy'):
            test_acc=total_acc.numpy()/batches
            test_acc=test_acc.astype(np.float32)
    else:
        if hasattr(platform,'DType'):
            batch_loss,batch_acc=test_tf(test_data,test_labels)
        else:
            batch_loss,batch_acc=test_pytorch(test_data,test_labels)
        test_loss=test_loss.numpy().astype(np.float32)
        if hasattr(nn,'accuracy'):
            test_acc=test_acc.numpy().astype(np.float32)
    print('test loss:{0:.6f}'.format(test_loss))
    if hasattr(nn,'accuracy'):
        if acc_flag=='%':
            print('test acc:{0:.1f}'.format(test_acc*100))
        else:
            print('test acc:{0:.6f}'.format(test_acc))
        if acc_flag=='%':
            return test_loss,test_acc*100
        else:
            return test_loss,test_acc
    else:
        return test_loss


class parallel_test:
    def __init__(self,nn,test_data,test_labels,process,batch,prefetch_batch_size=tf.data.AUTOTUNE,test_dataset=None,):
        self.nn=nn
        self.test_data=test_data
        self.test_labels=test_labels
        self.test_dataset=test_dataset
        self.process=process
        self.batch=batch
        self.loss=Array('f',np.zeros([process],dtype='float32'))
        if hasattr(nn,'accuracy'):
            self.acc=Array('f',np.zeros([process],dtype='float32'))
        self.prefetch_batch_size=prefetch_batch_size
    
    
    def segment_data(self):
        if len(self.test_data)!=self.process:
            data=np.array_split(self.test_data,self.process)
            labels=np.array_split(self.test_labels,self.process)
            self.test_data=data
            self.test_labels=labels
        return
    
    
    @tf.function(jit_compile=True)
    def test_(self,data,labels,p):
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
        try:
            if hasattr(self.nn,'accuracy'):
                try:
                    acc=self.nn.accuracy(output,labels,p)
                except Exception:
                    acc=self.nn.accuracy(output,labels)
            else:
                acc=None
        except Exception as e:
            raise e
        return loss,acc
    
    
    def test(self,p,training=None):
        if self.test_dataset is None:
            test_ds=tf.data.Dataset.from_tensor_slices((self.test_data[p],self.test_labels[p])).batch(self.batch).prefetch(self.prefetch_batch_size)
        elif self.test_dataset is not None and type(self.test_dataset)==list:
            test_ds=self.test_dataset[p]
        else:
            test_ds=self.test_dataset
        for data_batch,labels_batch in test_ds:
            if training!=None:
                training()
            try:
                batch_loss,batch_acc=self.test_(data_batch,labels_batch,p)
            except Exception as e:
                raise e
            if hasattr(self.nn,'accuracy'):
                self.loss[p]+=batch_loss
                self.acc[p]+=batch_acc
            else:
                self.loss[p]+=batch_loss
            if training!=None:
                training(False)
        return
    
    
    def loss_acc(self):
        if self.test_dataset is None:
            shape=len(self.test_data[0])*self.process
        elif self.test_dataset is not None and type(self.test_dataset)==list:
            shape=len(self.test_dataset[0])*len(self.test_dataset)
        else:
            shape=len(self.test_dataset)*self.process
        batches=int((shape-shape%self.batch)/self.batch)
        if shape%self.batch!=0:
            batches+=1
        if hasattr(self.nn,'accuracy'):
            return np.sum(npc.as_array(self.loss.get_obj()))/batches,np.sum(npc.as_array(self.acc.get_obj()))/batches
        else:
            return np.sum(npc.as_array(self.loss.get_obj()))/batches
