import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.batch_norm import batch_norm_
from Note.nn.layer.avg_pool2d import avg_pool2d
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.concat import concat
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Model import Model


class DenseLayer:
    def __init__(self,input_channels, growth_rate):
        self.layers=Layers()
        self.layers.add(batch_norm_(epsilon=1.001e-5))
        self.layers.add(activation_dict['relu'])
        self.layers.add(conv2d(4*growth_rate,[1,1],strides=1,padding="SAME",use_bias=False))
        self.layers.add(batch_norm_(epsilon=1.001e-5))
        self.layers.add(activation_dict['relu'])
        self.layers.add(conv2d(growth_rate,[3,3],strides=1,padding="SAME",use_bias=False))
        self.concat=concat()
        
        
    def __call__(self,x):
        x_=x
        x=self.layers(x)
        x=self.concat([x_,x])
        return x


def DenseBlock(input_channels, num_layers, growth_rate):
        layers=Layers()
        for i in range(num_layers):
            layers.add(DenseLayer(input_channels, growth_rate))
            input_channels=layers.output_size
        return layers


def TransitionLayer(input_channels, compression_factor):
        layers=Layers()
        layers.add(batch_norm_(input_channels))
        layers.add(activation_dict['relu'])
        layers.add(conv2d(int(compression_factor * input_channels),[1,1],strides=[1, 1, 1, 1],padding="SAME",use_bias=False))
        layers.add(avg_pool2d(ksize=[2, 2],strides=[2, 2],padding="SAME"))
        return layers


class DenseNet201(Model):
    def __init__(self, growth_rate=32, compression_factor=0.5, num_classes=1000, include_top=True, pooling=None, device='GPU'):
        super().__init__()
        self.num_classes=num_classes
        self.growth_rate=growth_rate
        self.compression_factor=compression_factor
        self.include_top=include_top
        self.pooling=pooling
        
        self.layers=Layers()
        self.layers.add(zeropadding2d(3,padding=[3, 3]))
        self.layers.add(conv2d(64,[7,7],strides=2,use_bias=False))
        self.layers.add(batch_norm_(epsilon=1.001e-5))
        self.layers.add(activation_dict['relu'])
        self.layers.add(zeropadding2d(padding=[1, 1]))
        self.layers.add(max_pool2d(ksize=[3, 3],strides=[2, 2],padding="VALID"))
        
        
        self.layers.add(DenseBlock(input_channels=64,num_layers=6,
                                 growth_rate=self.growth_rate))
        
        self.layers.add(TransitionLayer(input_channels=64,compression_factor=self.compression_factor))
        
        self.layers.add(DenseBlock(input_channels=int(64 * self.compression_factor),num_layers=12,
                                 growth_rate=self.growth_rate))
        
        self.layers.add(TransitionLayer(input_channels=int(64 * self.compression_factor),compression_factor=self.compression_factor))
        
        self.layers.add(DenseBlock(input_channels=int(64 * self.compression_factor**2),num_layers=48,
                                 growth_rate=self.growth_rate))
        
        self.layers.add(TransitionLayer(input_channels=int(64 * self.compression_factor**2),compression_factor=self.compression_factor))
        
        self.layers.add(DenseBlock(input_channels=int(64 * self.compression_factor**3),num_layers=32,
                                 growth_rate=self.growth_rate))
        
        self.layers.add(batch_norm_(epsilon=1.001e-5))
        
        self.layers.add(activation_dict['relu'])
        
        self.head=self.dense(self.num_classes,int(64 * self.compression_factor**3))
        
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer=Adam()
        self.device=device
        self.km=0
    
    
    def fp(self, data, p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x=self.layers(data)
                if self.include_top:
                    x = tf.reduce_mean(x, axis=[1, 2])
                    x = self.head(x)
                else:
                    if self.pooling == 'avg':
                        x = tf.reduce_mean(x, axis=[1, 2])
                    elif self.pooling == 'max':
                        x = tf.reduce_max(x, axis=[1, 2])
                return x
        else:
            x=self.layers(data,self.km)
            if self.include_top:
                x = tf.math.reduce_mean(x, axis=[1, 2])
                x = self.head(x)
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