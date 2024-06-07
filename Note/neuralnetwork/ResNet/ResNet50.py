import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.batch_norm import batch_norm_
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.identity import identity
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Model import Model


class block1:
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, conv_shortcut=True):
        self.layers1=Layers()
        if conv_shortcut:
            self.layers1.add(conv2d(4 * filters,[1,1],in_channels,strides=[stride]))
            self.layers1.add(batch_norm_(epsilon=1.001e-5))
        else:
            self.layers1.add(identity(in_channels))
        self.layers2=Layers()
        self.layers2.add(conv2d(filters,[1,1],in_channels,strides=[stride]))
        self.layers2.add(batch_norm_(epsilon=1.001e-5))
        self.layers2.add(activation_dict['relu'])
        self.layers2.add(conv2d(filters,[kernel_size,kernel_size],padding='SAME'))
        self.layers2.add(batch_norm_(epsilon=1.001e-5))
        self.layers2.add(activation_dict['relu'])
        self.layers2.add(conv2d(4 * filters,[1,1]))
        self.layers2.add(batch_norm_(epsilon=1.001e-5))
        self.train_flag=True
        self.output_size=self.layers2.output_size
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        shortcut=self.layers1(data,self.train_flag)
        x=self.layers2(data,self.train_flag)
        x=shortcut+x
        x=activation_dict['relu'](x)
        return x


def stack_fn(in_channels):
    layers=Layers()
    layers.add(stack1(in_channels, 64, 3, stride=1))
    layers.add(stack1(layers.output_size, 128, 4))
    layers.add(stack1(layers.output_size, 256, 6))
    layers.add(stack1(layers.output_size, 512, 3))
    return layers


def stack1(in_channels, filters, blocks, stride=2):
    layers=Layers()
    layers.add(block1(in_channels, filters, stride=stride))
    for i in range(2, blocks + 1):
        layers.add(block1(
            layers.output_size, filters, conv_shortcut=False
        ))
    return layers


class ResNet50(Model):
    def __init__(self,preact=False,classes=1000,include_top=True,pooling=None,use_bias=True,device='GPU'):
        super().__init__()
        self.preact=preact
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.use_bias=use_bias
        self.layers=Layers()
        self.layers.add(zeropadding2d(3,[3,3]))
        self.layers.add(conv2d(64,[7,7],strides=[2],use_bias=self.use_bias))
        if not self.preact:
            self.layers.add(batch_norm_(epsilon=1.001e-5))
            self.layers.add(activation_dict['relu'])
        self.layers.add(zeropadding2d(padding=[1,1]))
        self.layers.add(max_pool2d(ksize=[3, 3],strides=[2, 2],padding='SAME'))
        self.layers.add(stack_fn(self.layers.output_size))
        if self.preact:
            self.layers.add(batch_norm_(self.layers.output_size,epsilon=1.001e-5))
            self.layers.add(activation_dict['relu'])
        self.head=self.dense(self.classes,self.layers.output_size)
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer=Adam()
        self.km=0
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x=self.layers(data)
                if self.include_top:
                    x=tf.math.reduce_mean(x,axis=[1,2])
                    x=self.head(x)
                else:
                    if self.pooling=="avg":
                        x=tf.math.reduce_mean(x,axis=[1,2])
                    elif self.pooling=="max":
                        x=tf.math.reduce_max(x,axis=[1,2])
                return x
        else:
            x=self.layers(data,self.km)
            if self.include_top:
                x=tf.math.reduce_mean(x,axis=[1,2])
                x=self.head(x)
            else:
                if self.pooling=="avg":
                    x=tf.math.reduce_mean(x,axis=[1,2])
                elif self.pooling=="max":
                    x=tf.math.reduce_max(x,axis=[1,2])
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