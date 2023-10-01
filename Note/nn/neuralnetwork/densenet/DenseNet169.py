import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_normalization import batch_normalization
from Note.nn.layer.avg_pool2d import avg_pool2d
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Module import Module


def DenseLayer(data, growth_rate, train_flag, dtype='float32'):
        x=batch_normalization(epsilon=1.001e-5,dtype=dtype).output(data,train_flag)
        x=activation_dict['relu'](x)
        x=conv2d(4*growth_rate,[1,1],strides=1,padding="SAME",use_bias=False,dtype=dtype).output(x)
        x=batch_normalization(epsilon=1.001e-5,dtype=dtype).output(x,train_flag)
        x=activation_dict['relu'](data)
        x=conv2d(growth_rate,[3,3],strides=1,padding="SAME",use_bias=False,dtype=dtype).output(x)
        x=tf.concat([data,x],-1)
        return x


def DenseBlock(data, num_layers, growth_rate, train_flag=True, dtype='float32'):
        for i in range(num_layers):
            data=DenseLayer(data, growth_rate, train_flag, dtype)
        return data


def TransitionLayer(data, compression_factor, train_flag=True, dtype='float32'):
        x=batch_normalization(dtype=dtype).output(data,train_flag)
        x=activation_dict['relu'](x)
        x=conv2d(int(compression_factor * data.shape[-1]),[1,1],strides=[1, 1, 1, 1],padding="SAME",use_bias=False,dtype=dtype).output(x)
        x=avg_pool2d(ksize=[2, 2],strides=[2, 2],padding="SAME").output(x)
        return x


class DenseNet169:
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
        self.km=1
        self.param=Module.param
    
    
    def fp(self, data, p=None):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                x=zeropadding2d(padding=[3, 3]).output(data)
                x=conv2d(64,[7,7],strides=2,use_bias=False,dtype=self.dtype).output(x)
                x=batch_normalization(epsilon=1.001e-5,dtype=self.dtype).output(x)
                x=activation_dict['relu'](x)
                x=zeropadding2d(padding=[1, 1]).output(x)
                x=max_pool2d(ksize=[3, 3],strides=[2, 2],padding="VALID").output(x)
                x=DenseBlock(x,num_layers=6,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
                x=TransitionLayer(x,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
                x=DenseBlock(x,num_layers=12,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
                x=TransitionLayer(x,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
                x=DenseBlock(x,num_layers=32,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
                x=TransitionLayer(x,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
                x=DenseBlock(x,num_layers=32,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
                x=batch_normalization(epsilon=1.001e-5,dtype=self.dtype).output(x)
                x=activation_dict['relu'](x)
                if self.include_top:
                    x = tf.reduce_mean(x, axis=[1, 2])
                    x=dense(self.num_classes,activation='softmax',dtype=self.dtype).output(x)
                else:
                    if self.pooling == 'avg':
                        x = tf.reduce_mean(x, axis=[1, 2])
                    elif self.pooling == 'max':
                        x = tf.reduce_max(x, axis=[1, 2])
        else:
            x=zeropadding2d(padding=[3, 3]).output(data)
            x=conv2d(64,[7,7],strides=2,use_bias=False,dtype=self.dtype).output(x)
            x=batch_normalization(epsilon=1.001e-5,dtype=self.dtype).output(x,self.km)
            x=activation_dict['relu'](x)
            x=zeropadding2d(padding=[1, 1]).output(x)
            x=max_pool2d(ksize=[3, 3],strides=[2, 2],padding="VALID").output(x)
            x=DenseBlock(x,num_layers=6,
                             growth_rate=self.growth_rate,
                             train_flag=self.km,
                             dtype=self.dtype)
            x=TransitionLayer(x,compression_factor=self.compression_factor,train_flag=self.km,
                                  dtype=self.dtype)
            x=DenseBlock(x,num_layers=12,
                             growth_rate=self.growth_rate,
                             train_flag=self.km,
                             dtype=self.dtype)
            x=TransitionLayer(x,compression_factor=self.compression_factor,train_flag=self.km,
                                  dtype=self.dtype)
            x=DenseBlock(x,num_layers=32,
                             growth_rate=self.growth_rate,
                             train_flag=self.km,
                             dtype=self.dtype)
            x=TransitionLayer(x,compression_factor=self.compression_factor,train_flag=self.km,
                                  dtype=self.dtype)
            x=DenseBlock(x,num_layers=32,
                             growth_rate=self.growth_rate,
                             train_flag=self.km,
                             dtype=self.dtype)
            x=batch_normalization(epsilon=1.001e-5,dtype=self.dtype).output(x,self.km)
            x=activation_dict['relu'](x)
            x=dense(self.num_classes,dtype=self.dtype).output(x)
            if self.include_top:
                x = tf.math.reduce_mean(x, axis=[1, 2])
                x=dense(self.num_classes,activation='softmax',dtype=self.dtype).output(x)
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
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,'GPU')):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
        return tape,output,loss
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,'GPU')):
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param
