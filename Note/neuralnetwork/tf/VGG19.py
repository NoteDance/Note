import tensorflow as tf
from Note import nn


class VGG19(nn.Model):
    def __init__(self,include_top=True,pooling=None,classes=1000):
        super().__init__()
        self.include_top=include_top
        self.pooling=pooling
        self.classes=classes
        
        self.layers=nn.Sequential()
        # Block 1
        self.layers.add(nn.conv2d(64,(3,3),3,activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(64,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 2
        self.layers.add(nn.conv2d(128,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(128,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 3
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 4
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 5
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME"))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        
        self.flatten=nn.flatten
        self.dense1=nn.dense(4096,25088,activation='relu')
        self.dense2=nn.dense(4096,self.dense1.output_size,activation='relu')
        self.head=self.dense(self.classes,self.dense2.output_size)
    
    
    def __call__(self,data):
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