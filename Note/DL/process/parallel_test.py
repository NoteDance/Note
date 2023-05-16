import tensorflow as tf
import numpy as np
from multiprocessing import Array


class parallel_test:
    def __init__(self,nn,test_data,test_labels,process_t,process_num_t,batch):
        self.nn=nn
        self.test_data=test_data
        self.test_labels=test_labels
        self.process_t=process_t
        self.process_num_t=process_num_t
        self.batch=batch
        self.loss=Array('f',np.zeros([process_t],dtype=np.float32))
        try:
            if self.nn.accuracy!=None:
                self.acc=Array('f',np.zeros([process_t],dtype=np.float32))
        except AttributeError:
                pass
    
    
    def segment_data(self):
        if len(self.test_data)!=self.process_t:
            data=None
            labels=None
            segments=int((len(self.test_data)-len(self.test_data)%self.process_t)/self.process_t)
            for i in range(self.process_t):
                index1=i*segments
                index2=(i+1)*segments
                if i==0:
                    data=np.expand_dims(self.test_data[index1:index2],axis=0)
                    labels=np.expand_dims(self.test_labels[index1:index2],axis=0)
                else:
                    data=np.concatenate((data,np.expand_dims(self.test_data[index1:index2],axis=0)))
                    labels=np.concatenate((labels,np.expand_dims(self.test_labels[index1:index2],axis=0)))
            if len(data)%self.process_t!=0:
                segments+=1
                index1=segments*self.process_t
                index2=self.process_t-(len(self.train_data)-segments*self.process_t)
                data=np.concatenate((data,np.expand_dims(self.test_data[index1:index2],axis=0)))
                labels=np.concatenate((labels,np.expand_dims(self.test_labels[index1:index2],axis=0)))
            self.test_data=data
            self.test_labels=labels
        return
    
    
    def test(self):
        t=self.process_num_t.pop(0)
        if type(self.test_data)==list:
            train_ds=self.test_data[t]
        else:
            train_ds=tf.data.Dataset.from_tensor_slices((self.test_data[t],self.test_labels[t])).batch(self.batch)
        for data_batch,labels_batch in train_ds:
            try:
                try:
                    output=self.nn.fp(data_batch)
                    batch_loss=self.nn.loss(output,labels_batch)
                except TypeError:
                    output,batch_loss=self.nn.fp(data_batch,labels_batch)
            except TypeError:
                try:
                    output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                except TypeError:
                    output,batch_loss=self.nn.fp(data_batch,labels_batch,t) 
            try:
                if self.nn.accuracy!=None:
                    batch_acc=self.nn.accuracy(output,labels_batch)
            except AttributeError:
                pass
            try:
                if self.nn.accuracy!=None:
                    self.loss[t]+=batch_loss
                    self.acc[t]+=batch_acc
            except AttributeError:
                self.loss[t]+=batch_loss
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
                return np.mean(self.loss/batches),np.mean(self.acc/batches)
        except AttributeError:
            return np.mean(self.loss/batches)