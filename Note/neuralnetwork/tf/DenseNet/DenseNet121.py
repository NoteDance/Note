import tensorflow as tf
from Note import nn
from Note.nn.activation import activation_dict


def DenseLayer(input_channels, growth_rate, dtype='float32'):
        layers=nn.Layers()
        layers.add(nn.identity(input_channels),save_data=True)
        layers.add(nn.batch_norm_(epsilon=1.001e-5,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(nn.conv2d(4*growth_rate,[1,1],strides=1,padding="SAME",use_bias=False,dtype=dtype))
        layers.add(nn.batch_norm_(epsilon=1.001e-5,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(nn.conv2d(growth_rate,[3,3],strides=1,padding="SAME",use_bias=False,dtype=dtype),save_data=True)
        layers.add(nn.concat(),use_data=True)
        return layers


def DenseBlock(input_channels, num_layers, growth_rate, dtype='float32'):
        layers=nn.Layers()
        for i in range(num_layers):
            layers.add(DenseLayer(input_channels, growth_rate, dtype))
            input_channels=layers.output_size
        return layers


def TransitionLayer(input_channels, compression_factor, dtype='float32'):
        layers=nn.Layers()
        layers.add(nn.batch_norm_(input_channels,dtype=dtype))
        layers.add(activation_dict['relu'])
        layers.add(nn.conv2d(int(compression_factor * input_channels),[1,1],strides=[1, 1, 1, 1],padding="SAME",use_bias=False,dtype=dtype))
        layers.add(nn.avg_pool2d(ksize=[2, 2],strides=[2, 2],padding="SAME"))
        return layers


class DenseNet121:
    def __init__(self, growth_rate=32, compression_factor=0.5, num_classes=1000, include_top=True, pooling=None, dtype='float32'):
        self.num_classes=num_classes
        self.growth_rate=growth_rate
        self.compression_factor=compression_factor
        self.include_top=include_top
        self.pooling=pooling
        self.dtype=dtype
        self.training=True
    
    
    def build(self):
        nn.Model.init()
        
        self.layers=nn.Layers()
        self.layers.add(nn.zeropadding2d(3,padding=[3, 3]))
        self.layers.add(nn.conv2d(64,[7,7],strides=2,use_bias=False,dtype=self.dtype))
        self.layers.add(nn.batch_norm_(epsilon=1.001e-5,dtype=self.dtype))
        self.layers.add(activation_dict['relu'])
        self.layers.add(nn.zeropadding2d(padding=[1, 1]))
        self.layers.add(nn.max_pool2d(ksize=[3, 3],strides=[2, 2],padding="VALID"))
        
        
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
        
        self.layers.add(nn.batch_norm_(epsilon=1.001e-5,dtype=self.dtype))
        
        self.layers.add(activation_dict['relu'])
        
        self.dense=nn.dense(self.num_classes,self.layers.output_size,activation='softmax',dtype=self.dtype)
        
        self.param=nn.Model.param
        return
    
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.dense_=self.dense
            self.dense=nn.dense(classes,self.dense.input_size,activation='softmax',dtype=self.dense.dtype)
            param.extend(self.dense.param)
            self.param=param
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
    
    
    def fp(self, data):
        x=self.layers(data,self.training)
        if self.include_top:
            x = tf.math.reduce_mean(x, axis=[1, 2])
            x = self.dense(x)
        else:
            if self.pooling=="avg":
                x = tf.math.reduce_mean(x, axis=[1, 2])
            elif self.pooling=="max":
                x = tf.math.reduce_max(x, axis=[1, 2])
        return x