import tensorflow as tf
from Note import nn


class VGG19:
    def __init__(self,include_top=True,pooling=None,classes=1000):
        self.include_top=include_top
        self.pooling=pooling
        self.classes=classes
    
    
    def build(self,dtype='float32'):
        nn.Model.init()
        
        self.layers=nn.Layers()
        # Block 1
        self.layers.add(nn.conv2d(64,(3,3),3,activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(64,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 2
        self.layers.add(nn.conv2d(128,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(128,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 3
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 4
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        # Block 5
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype))
        self.layers.add(nn.max_pool2d((2, 2), strides=(2, 2), padding='VALID'))
        
        self.flatten=nn.flatten
        self.dense1=nn.dense(4096,25088,activation='relu',dtype=dtype)
        self.dense2=nn.dense(4096,self.dense1.output_size,activation='relu',dtype=dtype)
        self.dense3=nn.dense(self.classes,self.dense2.output_size,activation='softmax',dtype=dtype)
        
        self.param=nn.Model.param
        return
    
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.dense3_=self.dense3
            self.dense3=nn.dense(classes,self.dense3.input_size,activation='softmax',dtype=self.dense3.dtype)
            param.extend(self.dense1.param)
            self.param=param
            self.param.extend(self.dense2.param)
            self.param.extend(self.dense3.param)
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
    
    
    def __call__(self,data):
        x=self.layers(data)
        if self.include_top:
            x=self.flatten(x)
            x=self.dense1(x)
            x=self.dense2(x)
            x=self.dense3(x)
        else:
            if self.pooling=="avg":
                data = tf.math.reduce_mean(data, axis=[1, 2])
            elif self.pooling=="max":
                data = tf.math.reduce_max(data, axis=[1, 2])
        return x