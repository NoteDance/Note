import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.batch_normalization import batch_normalization
from Note.nn.layer.avg_pool2d import avg_pool2d
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.identity import identity
from Note.nn.layer.concat import concat
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.initializer import initializer
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


def DenseLayer(input_channels, growth_rate, dtype='float32'):
        layers=Layers()
        layers.add(identity(input_channels),save_data=True)
        layers.add(batch_normalization(keepdims=True,epsilon=1.001e-5,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(conv2d(4*growth_rate,[1,1],strides=1,padding="SAME",use_bias=False,dtype=dtype))
        layers.add(batch_normalization(keepdims=True,epsilon=1.001e-5,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(conv2d(growth_rate,[3,3],strides=1,padding="SAME",use_bias=False,dtype=dtype))
        layers.add(concat())
        return layers


def DenseBlock(input_channels, num_layers, growth_rate, dtype='float32'):
        layers=Layers()
        for i in range(num_layers):
            layers.add(DenseLayer(input_channels, growth_rate, dtype))
            input_channels=layers.output_size
        return layers


def TransitionLayer(input_channels, compression_factor, dtype='float32'):
        layers=Layers()
        layers.add(batch_normalization(input_channels,keepdims=True,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(conv2d(int(compression_factor * input_channels),[1,1],strides=[1, 1, 1, 1],padding="SAME",use_bias=False,dtype=dtype))
        layers.add(avg_pool2d(ksize=[2, 2],strides=[2, 2],padding="SAME"))
        return layers


class DenseNet121:
    def __init__(self, growth_rate=32, compression_factor=0.5, num_classes=1000, include_top=True, pooling=None, dtype='float32'):
        self.num_classes=num_classes
        self.growth_rate=growth_rate
        self.compression_factor=compression_factor
        self.include_top=include_top
        self.pooling=pooling
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.optimizer=Adam()
        self.bc=tf.Variable(0,dtype=dtype)
        self.dtype=dtype
        self.km=0
    
    
    def build(self):
        self.layers=Layers()
        self.layers.add(zeropadding2d(3,padding=[3, 3]))
        self.layers.add(conv2d(64,[7,7],strides=2,use_bias=False,dtype=self.dtype))
        self.layers.add(batch_normalization(keepdims=True,epsilon=1.001e-5,dtype=self.dtype))
        self.layers.add(activation_dict['relu'])
        self.layers.add(zeropadding2d(padding=[1, 1]))
        self.layers.add(max_pool2d(ksize=[3, 3],strides=[2, 2],padding="VALID"))
        
        
        self.layers.add(DenseBlock(input_channels=self.layers.output_size,num_layers=6,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype))
        
        self.layers.add(TransitionLayer(input_channels=self.layers.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype))
        
        self.layers.add(DenseBlock(input_channels=self.layers.output_size,num_layers=12,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype))
        
        self.layers.add(TransitionLayer(input_channels=self.layers.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype))
        
        self.layers.add(DenseBlock(input_channels=self.layers.output_size,num_layers=24,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype))
        
        self.layers.add(TransitionLayer(input_channels=self.layers.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype))
        
        self.layers.add(DenseBlock(input_channels=self.layers.output_size,num_layers=16,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype))
        
        self.layers.add(batch_normalization(keepdims=True,epsilon=1.001e-5,dtype=self.dtype))
        
        self.layers.add(activation_dict['relu'])
        
        self.fc_weight = initializer([self.layers.output_size, self.num_classes], 'Xavier', self.dtype)
        self.fc_bias = initializer([self.num_classes], 'Xavier', self.dtype)
        
        return
    
    
    def fp(self, data, p):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                x=self.layers.output(data)
                if self.include_top:
                    x = tf.reduce_mean(x, axis=[1, 2])
                    x = tf.matmul(x, self.fc_weight)+self.fc_bias
                    x = tf.nn.softmax(x)
                else:
                    if self.pooling == 'avg':
                        x = tf.reduce_mean(x, axis=[1, 2])
                    elif self.pooling == 'max':
                        x = tf.reduce_max(x, axis=[1, 2])
        else:
            x=self.layers.output(data)
            if self.include_top:
                x = tf.math.reduce_mean(x, axis=[1, 2])
                x = tf.matmul(x, self.fc_weight)+self.fc_bias
                x = tf.nn.softmax(x)
            else:
                if self.pooling=="avg":
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                elif self.pooling=="max":
                    x = tf.math.reduce_max(x, axis=[1, 2])
        return x
    
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,'GPU')):
            loss_value=self.loss_object(labels,output)
        return loss_value
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,'GPU')):
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param
