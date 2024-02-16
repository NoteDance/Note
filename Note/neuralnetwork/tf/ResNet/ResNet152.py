import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_norm import batch_norm
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.identity import identity
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.Module import Module


class block1:
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, conv_shortcut=True, dtype='float32'):
        self.layers1=Layers()
        if conv_shortcut:
            self.layers1.add(conv2d(4 * filters,[1,1],in_channels,strides=[stride],dtype=dtype))
            self.layers1.add(batch_norm(epsilon=1.001e-5,dtype=dtype))
        else:
            self.layers1.add(identity(in_channels))
        self.layers2=Layers()
        self.layers2.add(conv2d(filters,[1,1],in_channels,strides=[stride],dtype=dtype))
        self.layers2.add(batch_norm(epsilon=1.001e-5,dtype=dtype))
        self.layers2.add(activation_dict['relu'])
        self.layers2.add(conv2d(filters,[kernel_size,kernel_size],padding='SAME',dtype=dtype))
        self.layers2.add(batch_norm(epsilon=1.001e-5,dtype=dtype))
        self.layers2.add(activation_dict['relu'])
        self.layers2.add(conv2d(4 * filters,[1,1],dtype=dtype))
        self.layers2.add(batch_norm(epsilon=1.001e-5,dtype=dtype))
        self.train_flag=True
        self.output_size=self.layers2.output_size
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        shortcut=self.layers1(data,self.train_flag)
        x=self.layers2(data,self.train_flag)
        x=shortcut+x
        x=activation_dict['relu'](x)
        return x


def stack_fn(in_channels,dtype='float32'):
    layers=Layers()
    layers.add(stack1(in_channels, 64, 3, stride=1, dtype=dtype))
    layers.add(stack1(layers.output_size, 128, 8, dtype=dtype))
    layers.add(stack1(layers.output_size, 256, 36, dtype=dtype))
    layers.add(stack1(layers.output_size, 512, 3, dtype=dtype))
    return layers


def stack1(in_channels, filters, blocks, stride=2, dtype='float32'):
    layers=Layers()
    layers.add(block1(in_channels, filters, stride=stride, dtype=dtype))
    for i in range(2, blocks + 1):
        layers.add(block1(
            layers.output_size, filters, conv_shortcut=False, dtype=dtype
        ))
    return layers


class ResNet152:
    def __init__(self,preact=False,classes=1000,include_top=True,pooling=None,use_bias=True):
        self.preact=preact
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.use_bias=use_bias
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.training=True
    
    
    def build(self,dtype='float32'):
        Module.init()
        self.layers=Layers()
        self.layers.add(zeropadding2d(3,[3,3]))
        self.layers.add(conv2d(64,[7,7],strides=[2],use_bias=self.use_bias,dtype=dtype))
        if not self.preact:
            self.layers.add(batch_norm(epsilon=1.001e-5,dtype=dtype))
            self.layers.add(activation_dict['relu'])
        self.layers.add(zeropadding2d(padding=[1,1]))
        self.layers.add(max_pool2d(ksize=[3, 3],strides=[2, 2],padding='SAME'))
        self.layers.add(stack_fn(self.layers.output_size,dtype=dtype))
        if self.preact:
            self.layers.add(batch_norm(self.layers.output_size,epsilon=1.001e-5,dtype=dtype))
            self.layers.add(activation_dict['relu'])
        self.dense=dense(self.classes,self.layers.output_size,activation='softmax',dtype=dtype)
        self.param=Module.param
        return
    
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
            self.dense_=self.dense
            self.dense=dense(classes,self.dense.input_size,activation='softmax',dtype=self.dense.dtype)
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
    
    
    def __call__(self,data):
        x=self.layers(data,self.training)
        if self.include_top:
            x=tf.math.reduce_mean(x,axis=[1,2])
            x=self.dense(x)
        else:
            if self.pooling=="avg":
                x=tf.math.reduce_mean(x,axis=[1,2])
            elif self.pooling=="max":
                x=tf.math.reduce_max(x,axis=[1,2])
        return x