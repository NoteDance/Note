import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.flatten import flatten 
from Note.nn.Sequential import Sequential
from Note.nn.Model import Model


class VGG19(Model):
    def __init__(self,include_top=True,pooling=None,classes=1000):
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
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 4
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 5
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        
        self.flatten=flatten
        self.dense1=dense(4096,25088,activation='relu')
        self.dense2=dense(4096,self.dense1.output_size,activation='relu')
        self.head=self.dense(self.classes,self.dense2.output_size)
        
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.opt=tf.keras.optimizers.Adam()
        self.km=0
    
    
    def fp(self,data):
        x=self.layers(data)
        if self.include_top:
            x=self.flatten(x)
            x=self.dense1(x)
            x=self.dense2(x)
            x=self.head(x)
        else:
            if self.pooling=="avg":
                data = tf.math.reduce_mean(data, axis=[1, 2])
            elif self.pooling=="max":
                data = tf.math.reduce_max(data, axis=[1, 2])
        return x


    def loss(self,output,labels):
        loss=self.loss_object(labels,output)
        return loss