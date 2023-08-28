import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_normalization import batch_normalization
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class block1:
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, conv_shortcut=True, dtype='float32'):
        self.conv2d1=conv2d([1,1,in_channels,4 * filters],strides=[stride],dtype=dtype)
        self.batch_norm1=batch_normalization(self.conv2d1.output_size,epsilon=1.001e-5,keepdims=True,dtype=dtype)
        self.conv2d2=conv2d([1,1,in_channels,filters],strides=[stride],dtype=dtype)
        self.batch_norm2=batch_normalization(self.conv2d2.output_size,epsilon=1.001e-5,keepdims=True,dtype=dtype)
        self.conv2d3=conv2d([kernel_size,kernel_size,self.conv2d2.output_size,filters],padding='SAME',dtype=dtype)
        self.batch_norm3=batch_normalization(self.conv2d3.output_size,epsilon=1.001e-5,keepdims=True,dtype=dtype)
        self.conv2d4=conv2d([1,1,self.conv2d3.output_size,4 * filters],dtype=dtype)
        self.batch_norm4=batch_normalization(self.conv2d4.output_size,epsilon=1.001e-5,keepdims=True,dtype=dtype)
        self.conv_shortcut=conv_shortcut
        self.train_flag=True
        self.output_size=self.conv2d4.output_size
        self.param=[self.conv2d1.param,self.batch_norm1.param,self.conv2d2.param,self.batch_norm2.param,
                    self.conv2d3.param,self.batch_norm3.param,self.conv2d4.param,self.batch_norm4.param]
    
    
    def output(self,data,train_flag=True):
        self.train_flag=train_flag
        if self.conv_shortcut:
            shortcut=self.conv2d1.output(data)
            shortcut==self.batch_norm1.output(shortcut,train_flag)
        else:
            shortcut = data
        x=self.conv2d2.output(data)
        x=self.batch_norm2.output(x,train_flag)
        x=activation_dict['relu'](x)
        x=self.conv2d3.output(x)
        x=self.batch_norm3.output(x,train_flag)
        x=activation_dict['relu'](x)
        x=self.conv2d4.output(x)
        x=self.batch_norm4.output(x,train_flag)
        x=shortcut+x
        x=activation_dict['relu'](x)
        return x


def stack_fn(in_channels,dtype='float32'):
    layers=Layers()
    layers.add(stack1(in_channels, 64, 3, stride1=1, dtype=dtype))
    layers.add(stack1(layers.output_size, 128, 4, dtype=dtype))
    layers.add(stack1(layers.output_size, 256, 23, dtype=dtype))
    layers.add(stack1(layers.output_size, 512, 3, dtype=dtype))
    return layers


def stack1(in_channels, filters, blocks, stride1=2, dtype='float32'):
    layers=Layers()
    layers.add(block1(in_channels, filters, stride=stride1, dtype=dtype))
    for i in range(2, blocks + 1):
        layers.add(block1(
            layers.output_size, filters, conv_shortcut=False, dtype=dtype
        ))
    return layers


class ResNet101:
    def __init__(self,preact=False,classes=1000,include_top=True,pooling=None,use_bias=True):
        self.preact=preact
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.use_bias=use_bias
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.optimizer=Adam()
        self.param=[]
        self.km=0
    
    
    def build(self,dtype='float32'):
        self.zeropadding2d1=tf.pad
        self.conv2d1=conv2d([7,7,3,64],strides=[2],use_bias=self.use_bias,dtype=dtype)
        if not self.preact:
            self.batch_norm1=batch_normalization(self.conv2d1.output_size,epsilon=1.001e-5,keepdims=True,dtype=dtype)
            self.param.append(self.batch_norm1.param)
        self.zeropadding2d2=tf.pad
        self.maxpooling2d=tf.nn.max_pool2d
        self.stack=stack_fn(self.conv2d1.output_size,dtype=dtype)
        if self.preact:
            self.batch_norm2=batch_normalization(self.stack.output_size,epsilon=1.001e-5,keepdims=True,dtype=dtype)
            self.param.append(self.batch_norm2.param)
        self.dense=dense([self.stack.output_size,self.classes],activation='softmax',dtype=dtype)
        self.param.extend([self.conv2d1.param,self.stack.param,self.dense.param])
        return
    
    
    def fp(self,data,p):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                x=self.zeropadding2d1(data, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
                x=self.conv2d1.output(x)
                if not self.preact:
                    x=self.batch_norm1.output(x)
                    x=activation_dict['relu'](x)
                x=self.zeropadding2d2(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
                x=self.maxpooling2d(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                x=self.stack.output(x)
                if self.preact:
                    x=self.batch_norm1.output(x)
                    x=activation_dict['relu'](x)
                if self.include_top:
                    x=tf.math.reduce_mean(x,axis=[1,2])
                    x=self.dense.output(x)
                else:
                    if self.pooling=="avg":
                        x=tf.math.reduce_mean(x,axis=[1,2])
                    elif self.pooling=="max":
                        x=tf.math.reduce_max(x,axis=[1,2])
        else:
            x=self.zeropadding2d1(data, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
            x=self.conv2d1.output(x)
            if not self.preact:
                x=activation_dict['relu'](x)
            x=self.zeropadding2d2(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            x=self.maxpooling2d(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            x=self.stack.output(x,self.km)
            if self.preact:
                x=activation_dict['relu'](x)
            if self.include_top:
                x=tf.math.reduce_mean(x,axis=[1,2])
                x=self.dense.output(x)
            else:
                if self.pooling=="avg":
                    x=tf.math.reduce_mean(x,axis=[1,2])
                elif self.pooling=="max":
                    x=tf.math.reduce_max(x,axis=[1,2])
        return x
    
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,'GPU')):
            loss_value=self.loss_object(labels,output)
        return loss_value
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,'GPU')):
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param
