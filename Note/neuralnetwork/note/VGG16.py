import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.flatten import flatten 
from Note.nn.Sequential import Sequential
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Model import Model


class VGG16(Model):
    def __init__(self,include_top=True,pooling=None,classes=1000,device='GPU'):
        super().__init__()
        self.include_top=include_top
        self.pooling=pooling
        self.classes=classes
        
        self.layers=Sequential()
        # Block 1
        self.layers.add(conv2d(64,(3,3),3,activation="relu", padding="SAME"))
        self.layers.add(conv2d(64,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 2
        self.layers.add(conv2d(128,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(128,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 3
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 4
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 5
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        
        self.flatten=flatten()
        self.dense1=dense(4096,25088,activation='relu')
        self.dense2=dense(4096,self.dense1.output_size,activation='relu')
        self.head=self.dense(self.classes,self.dense2.output_size,activation='softmax')
        
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer=Adam()
        self.km=0
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x=self.layers(data)
                if self.include_top:
                    x=self.flatten(x)
                    x=self.dense1(x)
                    x=self.dense2(x)
                    x=self.head(x)
                else:
                    if self.pooling=="avg":
                        x = tf.math.reduce_mean(x, axis=[1, 2])
                    elif self.pooling=="max":
                        x = tf.math.reduce_max(x, axis=[1, 2])
                return x
        else:
            x=self.layers(data)
            if self.include_top:
                x=self.flatten(x)
                x=self.dense1(x)
                x=self.dense2(x)
                x=self.head(x)
            else:
                if self.pooling=="avg":
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                elif self.pooling=="max":
                    x = tf.math.reduce_max(x, axis=[1, 2])
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
            param=self.optimizer(gradient,self.param,self.bc[0])
            return param
