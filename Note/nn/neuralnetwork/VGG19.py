import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.flatten import flatten 
from Note.nn.Layers import Layers
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Module import Module


class VGG19:
    def __init__(self,include_top=True,pooling=None,classes=1000,device='GPU'):
        self.include_top=include_top
        self.pooling=pooling
        self.classes=classes
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.km=0
    
    
    def build(self,dtype='float32'):
        self.layers=Layers()
        # Block 1
        self.layers.add(conv2d(64,(3,3),3,activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(64,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 2
        self.layers.add(conv2d(128,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(128,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 3
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 4
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 5
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        
        self.flatten=flatten
        self.dense1=dense(4096,25088,activation='relu',dtype=dtype)
        self.dense2=dense(4096,self.dense1.output_size,activation='relu',dtype=dtype)
        self.dense3=dense(self.classes,self.dense2.output_size,activation='softmax',dtype=dtype)
        
        self.optimizer=Adam()
        self.param=Module.param
        return
    
    
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
            self.dense3_=self.dense3
            self.dense3=dense(classes,self.dense3.input_size,activation='softmax',dtype=self.dense3.dtype)
            param.extend(self.dense1.param)
            self.param=param
            self.param.extend(self.dense2.param)
            self.param.extend(self.dense3.param)
            self.optimizer_=self.optimizer
            self.optimizer=Adam(lr=lr,param=self.param)
        elif flag==1:
            del self.param_[-len(self.dense3.param):]
            self.param_.extend(self.dense3.param)
            self.param=self.param_
        else:
            self.dense3,self.dense3_=self.dense3_,self.dense3
            del self.param_[-len(self.dense3.param):]
            self.param_.extend(self.dense3.param)
            self.param=self.param_
        return
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x=self.layers.output(data)
                if self.include_top:
                    x=self.flatten(x)
                    x=self.dense1.output(x)
                    x=self.dense2.output(x)
                    x=self.dense3.output(x)
                else:
                    if self.pooling=="avg":
                        x = tf.math.reduce_mean(x, axis=[1, 2])
                    elif self.pooling=="max":
                        x = tf.math.reduce_max(x, axis=[1, 2])
        else:
            x=self.layers.output(data)
            if self.include_top:
                x=self.flatten(x)
                x=self.dense1.output(x)
                x=self.dense2.output(x)
                x=self.dense3.output(x)
            else:
                if self.pooling=="avg":
                    data = tf.math.reduce_mean(data, axis=[1, 2])
                elif self.pooling=="max":
                    data = tf.math.reduce_max(data, axis=[1, 2])
        return x


    def loss(self,output,labels,p):
        with tf.device(assign_device(p,self.device)):
            loss=self.loss_object(labels,output)
        return loss
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
        return tape,output,loss
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,self.device)):
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param
