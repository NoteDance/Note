import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.flatten import flatten 
from Note.nn.Layers import Layers
from Note.nn.parallel.optimizer_ import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Module import Module


class VGG16:
    def __init__(self,include_top=True,pooling=None,classes=1000,device='GPU'):
        """
        This is an example for demonstration, because it uses the parallel optimizer in the 
        Note.nn.parallel.optimizer_ module, so it can determine the shape of the input for each layer 
        and initialize the parameters during the forward propagation, which is convenient for 
        building neural networks. However, the optimization effect of using the parallel optimizer in 
        the Note.nn.parallel.optimizer_ module may be worse than the optimization effect of using 
        the parallel optimizer in the Note.nn.parallel.optimizer module.
        """
        self.include_top=include_top
        self.pooling=pooling
        self.classes=classes
        self.device=device
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.km=0
    
    
    def build(self,dtype='float32'):
        self.layers=Layers()
        # Block 1
        self.conv2d1=conv2d(64,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.conv2d2=conv2d(64,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.max_pool2d1=max_pool2d((2, 2), strides=(2, 2), padding='VALID')
        # Block 2
        self.conv2d3=conv2d(128,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.conv2d4=conv2d(128,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.max_pool2d2=max_pool2d((2, 2), strides=(2, 2), padding='VALID')
        # Block 3
        self.conv2d4=conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.conv2d5=conv2d(256,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.max_pool2d3=max_pool2d((2, 2), strides=(2, 2), padding='VALID')
        # Block 4
        self.conv2d6=conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.conv2d7=conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.conv2d8=conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.max_pool2d4=max_pool2d((2, 2), strides=(2, 2), padding='VALID')
        # Block 5
        self.conv2d9=conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.conv2d10=conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.conv2d11=conv2d(512,(3,3),activation="relu", padding="SAME",dtype=dtype)
        self.max_pool2d5=max_pool2d((2, 2), strides=(2, 2), padding='VALID')
        
        self.flatten=flatten()
        self.dense1=dense(4096,activation='relu',dtype=dtype)
        self.dense2=dense(4096,activation='relu',dtype=dtype)
        self.dense3=dense(self.classes,activation='softmax',dtype=dtype)
        
        self.optimizer=Adam()
        self.param=Module.param
        return
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                # Block 1
                x=self.conv2d1.output(data)
                x=self.conv2d2.output(x)
                x=self.max_pool2d1.output(x)
                # Block 2
                x=self.conv2d3.output(x)
                x=self.conv2d4.output(x)
                x=self.max_pool2d2.output(x)
                # Block 3
                x=self.conv2d4.output(x)
                x=self.conv2d5.output(x)
                x=self.max_pool2d3.output(x)
                # Block 4
                x=self.conv2d6.output(x)
                x=self.conv2d7.output(x)
                x=self.conv2d8.output(x)
                x=self.max_pool2d4.output(x)
                # Block 5
                x=self.conv2d9.output(x)
                x=self.conv2d10.output(x)
                x=self.conv2d11.output(x)
                x=self.max_pool2d5.output(x)
                if self.include_top:
                    x=self.flatten.output(x)
                    x=self.dense1.output(x)
                    x=self.dense2.output(x)
                    x=self.dense3.output(x)
                else:
                    if self.pooling=="avg":
                        x = tf.math.reduce_mean(x, axis=[1, 2])
                    elif self.pooling=="max":
                        x = tf.math.reduce_max(x, axis=[1, 2])
        else:
            # Block 1
            x=self.conv2d1.output(data)
            x=self.conv2d2.output(x)
            x=self.max_pool2d1.output(x)
            # Block 2
            x=self.conv2d3.output(x)
            x=self.conv2d4.output(x)
            x=self.max_pool2d2.output(x)
            # Block 3
            x=self.conv2d4.output(x)
            x=self.conv2d5.output(x)
            x=self.max_pool2d3.output(x)
            # Block 4
            x=self.conv2d6.output(x)
            x=self.conv2d7.output(x)
            x=self.conv2d8.output(x)
            x=self.max_pool2d4.output(x)
            # Block 5
            x=self.conv2d9.output(x)
            x=self.conv2d10.output(x)
            x=self.conv2d11.output(x)
            x=self.max_pool2d5.output(x)
            if self.include_top:
                x=self.flatten.output(x)
                x=self.dense1.output(x)
                x=self.dense2.output(x)
                x=self.dense3.output(x)
            else:
                if self.pooling=="avg":
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                elif self.pooling=="max":
                    x = tf.math.reduce_max(x, axis=[1, 2])
        return x


    def loss(self,output,labels,p):
        with tf.device(assign_device(p,self.device)):
            loss_value=self.loss_object(labels,output)
        return loss_value
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
        return tape,output,loss
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,self.device)):
            param=self.optimizer.opt(gradient,self.param)
            return param