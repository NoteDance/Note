import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_norm import batch_norm
from Note.nn.layer.avg_pool2d import avg_pool2d
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.identity import identity
from Note.nn.layer.concat import concat
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Module import Module


def DenseLayer(input_channels, growth_rate, dtype='float32'):
        layers=Layers()
        layers.add(identity(input_channels),save_data=True)
        layers.add(batch_norm(epsilon=1.001e-5,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(conv2d(4*growth_rate,[1,1],strides=1,padding="SAME",use_bias=False,dtype=dtype))
        layers.add(batch_norm(epsilon=1.001e-5,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(conv2d(growth_rate,[3,3],strides=1,padding="SAME",use_bias=False,dtype=dtype),save_data=True)
        layers.add(concat(),use_data=True)
        return layers


def DenseBlock(input_channels, num_layers, growth_rate, dtype='float32'):
        layers=Layers()
        for i in range(num_layers):
            layers.add(DenseLayer(input_channels, growth_rate, dtype))
            input_channels=layers.output_size
        return layers


def TransitionLayer(input_channels, compression_factor, dtype='float32'):
        layers=Layers()
        layers.add(batch_norm(input_channels,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(conv2d(int(compression_factor * input_channels),[1,1],strides=[1, 1, 1, 1],padding="SAME",use_bias=False,dtype=dtype))
        layers.add(avg_pool2d(ksize=[2, 2],strides=[2, 2],padding="SAME"))
        return layers


class DenseNet121:
    def __init__(self, growth_rate=32, compression_factor=0.5, num_classes=1000, include_top=True, pooling=None, device='GPU', dtype='float32'):
        self.num_classes=num_classes
        self.growth_rate=growth_rate
        self.compression_factor=compression_factor
        self.include_top=include_top
        self.pooling=pooling
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.device=device
        self.dtype=dtype
        self.km=0
    
    
    def build(self):
        Module.init()
        
        self.layers=Layers()
        self.layers.add(zeropadding2d(3,padding=[3, 3]))
        self.layers.add(conv2d(64,[7,7],strides=2,use_bias=False,dtype=self.dtype))
        self.layers.add(batch_norm(epsilon=1.001e-5,dtype=self.dtype))
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
        
        self.layers.add(batch_norm(epsilon=1.001e-5,dtype=self.dtype))
        
        self.layers.add(activation_dict['relu'])
        
        self.dense=dense(self.num_classes,self.layers.output_size,activation='softmax',dtype=self.dtype)
        
        self.optimizer=Adam()
        self.param=Module.param
        return
    
    
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
            self.dense_=self.dense
            self.dense=dense(classes,self.dense.input_size,activation='softmax',dtype=self.dense.dtype)
            param.extend(self.dense.param)
            self.param=param
            self.optimizer_=self.optimizer
            self.optimizer=Adam(lr=lr,param=self.param)
        elif flag==1:
            del self.param_[-len(self.dense.param):]
            self.param_.extend(self.dense.param)
            self.param=self.param_
        else:
            self.dense,self.dense_=self.dense_,self.dense
            del self.param_[-len(self.dense.param):]
            self.param_.extend(self.dense.param)
            self.param=self.param_
        return
    
    
    def fp(self, data, p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x=self.layers.output(data)
                if self.include_top:
                    x = tf.reduce_mean(x, axis=[1, 2])
                    x = self.dense.output(x)
                else:
                    if self.pooling == 'avg':
                        x = tf.reduce_mean(x, axis=[1, 2])
                    elif self.pooling == 'max':
                        x = tf.reduce_max(x, axis=[1, 2])
        else:
            x=self.layers.output(data,self.km)
            if self.include_top:
                x = tf.math.reduce_mean(x, axis=[1, 2])
                x = self.dense.output(x)
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
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param