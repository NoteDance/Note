import tensorflow as tf
import Note.create.create as c
from tensorflow.python.ops import state_ops


class unnamed:
    def __init__():
        with tf.name_scope('data'):
            
            
        with tf.name_scope('parameter'):
            
            
            self.parameter=[]
        with tf.name_scope('hyperparameter'):
            self.hyperparameter=None
        with tf.name_scope('regulation'):
            self.regulation=None
        with tf.name_scope('optimizer'):
            self.opt=None
        self.p_accumulator=0
        self.acc_flag1=1
        self.acc_flag2=None
        self.flag=0
    
    
    def weight_init(self,shape,mean,stddev):
        if self.flag==1:
            self.p_accumulator+=1
            return self.parameter[self.p_accumulator-1]
        else:
            self.parameter.append(tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype)))
            return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
    
    
    def bias_init(self,shape,mean,stddev):
        if self.flag==1:
            self.p_accumulator+=1
            return self.parameter[self.p_accumulator-1]
        else:
            self.parameter.append(tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype)))
            return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype))
    
    
    def structure(self):
        self.p_accumulator=0
        self.dtype=dtype
        with tf.name_scope('hyperparameter'):
            
            
        with tf.name_scope('parameter_initialization'):
            
            
            
    @tf.function       
    def forward_propagation(self,train_data,dropout=None):
        with tf.name_scope('processor_allocation'):
            
            
        with tf.name_scope('forward_propagation'):
            
            
    
    def loss(self,output,train_labels,l2=None):
        
        
    
    def accuracy(self,output,train_labels):
        
