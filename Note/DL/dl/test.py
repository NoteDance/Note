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
        acc=nn.accuracy(output,labels)
    except Exception as e:
        try:
            if nn.accuracy!=None:
                raise e
        except Exception:
            acc=None
            pass
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
        acc=nn.accuracy(output,labels)
    except Exception as e:
        try:
            if nn.accuracy!=None:
                raise e
        except Exception:
            acc=None
            pass
    return loss,acc


def test(nn,test_data,test_labels,platform,batch=None,loss=None,acc_flag='%'):
    if type(nn.param[0])!=list:
        test_data=test_data.astype(nn.param[0].dtype.name)
        test_labels=test_labels.astype(nn.param[0].dtype.name)
    else:
        test_data=test_data.astype(nn.param[0][0].dtype.name)
        test_labels=test_labels.astype(nn.param[0][0].dtype.name)
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
            try:
                try:
                    if platform.DType!=None:
                        batch_loss,batch_acc=test_tf(data_batch,labels_batch)
                except Exception:
                    batch_loss,batch_acc=test_pytorch(data_batch,labels_batch)
            except Exception as e:
                raise e
            total_loss+=batch_loss
            try:
                total_acc+=batch_acc
            except Exception:
                pass
        if shape0%batch!=0:
            batches+=1
            index1=batches*batch
            index2=batch-(shape0-batches*batch)
            try:
                try:
                    data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)
                    labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)
                except Exception:
                    data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)
                    labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)
            except Exception as e:
                raise e
            try:
                try:
                    if platform.DType!=None:
                        batch_loss,batch_acc=test_tf(data_batch,labels_batch)
                except Exception:
                    batch_loss,batch_acc=test_pytorch(data_batch,labels_batch)
            except Exception as e:
                raise e
            total_loss+=batch_loss
            try:
                total_acc+=batch_acc
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
                if platform.DType!=None:
                    batch_loss,batch_acc=test_tf(test_data,test_labels)
            except Exception:
                batch_loss,batch_acc=test_pytorch(test_data,test_labels)
        except Exception as e:
            raise e
        test_loss=test_loss.numpy().astype(np.float32)
        try:
            test_acc=test_acc.numpy().astype(np.float32)
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
    def __init__(self,nn,test_data,test_labels,process,batch,prefetch_batch_size=tf.data.AUTOTUNE):
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
        self.prefetch_batch_size=prefetch_batch_size
    
    
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
    
    
    @tf.function(jit_compile=True)
    def test_(self,data,labels,p):
        try:
            try:
                try:
                    output=self.nn.fp(data)
                    loss=self.nn.loss(output,labels)
                except Exception:
                    output,loss=self.nn.fp(data,labels)
            except Exception:
                try:
                    output=self.nn.fp(data,p)
                    loss=self.nn.loss(output,labels)
                except Exception:
                    output,loss=self.nn.fp(data,labels,p)
        except Exception as e:
            raise e
        try:
            acc=self.nn.accuracy(output,labels)
        except Exception as e:
            try:
                if self.nn.accuracy!=None:
                    raise e
            except Exception:
                acc=None
                pass
        return loss,acc
    
    
    def test(self):
        p=self.process_num.pop(0)
        if type(self.test_data)==list:
            train_ds=self.test_data[p]
        else:
            train_ds=tf.data.Dataset.from_tensor_slices((self.test_data[p],self.test_labels[p])).batch(self.batch).prefetch(self.prefetch_batch_size)
        for data_batch,labels_batch in train_ds:
            try:
                batch_loss,batch_acc=self.test_(data_batch,labels_batch,p)
            except Exception as e:
                raise e
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
