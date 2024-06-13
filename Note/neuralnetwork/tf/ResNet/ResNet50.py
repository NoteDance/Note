import tensorflow as tf
from Note import nn
from Note.nn.activation import activation_dict


class block1:
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, conv_shortcut=True):
        self.layers1=nn.Sequential()
        if conv_shortcut:
            self.layers1.add(nn.conv2d(4 * filters,[1,1],in_channels,strides=[stride]))
            self.layers1.add(nn.batch_norm(epsilon=1.001e-5))
        else:
            self.layers1.add(nn.identity(in_channels))
        self.layers2=nn.Sequential()
        self.layers2.add(nn.conv2d(filters,[1,1],in_channels,strides=[stride]))
        self.layers2.add(nn.batch_norm(epsilon=1.001e-5))
        self.layers2.add(activation_dict['relu'])
        self.layers2.add(nn.conv2d(filters,[kernel_size,kernel_size],padding='SAME'))
        self.layers2.add(nn.batch_norm(epsilon=1.001e-5))
        self.layers2.add(activation_dict['relu'])
        self.layers2.add(nn.conv2d(4 * filters,[1,1]))
        self.layers2.add(nn.batch_norm(epsilon=1.001e-5))
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
    layers=nn.Sequential()
    layers.add(stack1(in_channels, 64, 3, stride=1))
    layers.add(stack1(layers.output_size, 128, 4))
    layers.add(stack1(layers.output_size, 256, 6))
    layers.add(stack1(layers.output_size, 512, 3))
    return layers


def stack1(in_channels, filters, blocks, stride=2):
    layers=nn.Sequential()
    layers.add(block1(in_channels, filters, stride=stride))
    for i in range(2, blocks + 1):
        layers.add(block1(
            layers.output_size, filters, conv_shortcut=False
        ))
    return layers


class ResNet50(nn.Model):
    def __init__(self,preact=False,classes=1000,include_top=True,pooling=None,use_bias=True):
        super().__init__()
        self.preact=preact
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.use_bias=use_bias
        self.layers=nn.Sequential()
        self.layers.add(nn.zeropadding2d(3,[3,3]))
        self.layers.add(nn.conv2d(64,[7,7],strides=[2],use_bias=self.use_bias))
        if not self.preact:
            self.layers.add(nn.batch_norm(epsilon=1.001e-5))
            self.layers.add(activation_dict['relu'])
        self.layers.add(nn.zeropadding2d(padding=[1,1]))
        self.layers.add(nn.max_pool2d(ksize=[3, 3],strides=[2, 2],padding='SAME'))
        self.layers.add(stack_fn(self.layers.output_size))
        if self.preact:
            self.layers.add(nn.batch_norm(self.layers.output_size,epsilon=1.001e-5))
            self.layers.add(activation_dict['relu'])
        self.head=self.dense(self.classes,self.layers.output_size)
        self.training=True
    
    
    def __call__(self,data):
        x=self.layers(data,self.training)
        if self.include_top:
            x=tf.math.reduce_mean(x,axis=[1,2])
            x=self.head(x)
        else:
            if self.pooling=="avg":
                x=tf.math.reduce_mean(x,axis=[1,2])
            elif self.pooling=="max":
                x=tf.math.reduce_max(x,axis=[1,2])
        return x