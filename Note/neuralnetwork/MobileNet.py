import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.batch_norm_ import batch_norm_
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Model import Model


class _conv_block:
    def __init__(self, in_channels, filters, alpha, kernel=(3, 3), strides=[1, 1]):
        filters = int(filters * alpha)
        self.conv2d=conv2d(filters,kernel,in_channels,strides=strides,padding='SAME',use_bias=False)
        self.batch_norm_=batch_norm_(self.conv2d.output_size)
        self.train_flag=True
        self.output_size=self.conv2d.output_size
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        x=self.conv2d(data)
        x=self.batch_norm_(x,self.train_flag)
        return activation_dict['relu6'](x)


class _depthwise_conv_block:
    def __init__(self,in_channels,pointwise_conv_filters,alpha,depth_multiplier=1,strides=[1, 1],block_id=1):
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)
        self.strides=strides
        self.zeropadding2d=zeropadding2d()
        self.depthwiseconv2d=depthwise_conv2d([3,3],depth_multiplier,in_channels,strides=[1,strides[0],strides[1],1],padding="SAME" if strides == [1, 1] else "VALID",use_bias=False)
        self.batch_norm1=batch_norm_(self.depthwiseconv2d.output_size)
        self.conv2d=conv2d(pointwise_conv_filters,[1,1],self.depthwiseconv2d.output_size,strides=[1, 1],padding='SAME',use_bias=False)
        self.batch_norm2=batch_norm_(self.conv2d.output_size)
        self.train_flag=True
        self.output_size=self.conv2d.output_size
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        if self.strides == [1, 1]:
            x = data
        else:
            x = self.zeropadding2d(data, ((0, 1), (0, 1)))
        x=self.depthwiseconv2d(x)
        x=self.batch_norm1(x,self.train_flag)
        x=activation_dict['relu6'](x)
        x=self.conv2d(x)
        x=self.batch_norm2(x,self.train_flag)
        return activation_dict['relu6'](x)


class MobileNet(Model):
    def __init__(self, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, pooling=None, classes=1000, device='GPU'):
        super().__init__()
        self.alpha=alpha
        self.depth_multiplier=depth_multiplier
        self.dropout=dropout
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        
        self.layers=Layers()
        self.layers.add(_conv_block(3, 32, self.alpha, strides=[2, 2]))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 64, self.alpha, self.depth_multiplier, block_id=1))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 128, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=2
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 128, self.alpha, self.depth_multiplier, block_id=3))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 256, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=4
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 256, self.alpha, self.depth_multiplier, block_id=5))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 512, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=6
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=7))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=8))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=9))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=10))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=11))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 1024, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=12
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 1024, self.alpha, self.depth_multiplier, block_id=13))
        self.conv2d=conv2d(self.classes,[1,1],self.layers.output_size,padding='SAME')
        self.head=self.dense(self.classes,self.conv2d.output_size)
        
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer=Adam()
        self.km=0
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x=self.layers(data)
                if self.include_top:
                    x=tf.math.reduce_mean(x,axis=[1,2],keepdims=True)
                    x=tf.nn.dropout(x,self.dropout)
                    x=self.conv2d(x)
                    x=tf.reshape(x,[x.shape[0],self.classes])
                    x=self.head(x)
                else:
                    if self.pooling == "avg":
                        x=tf.math.reduce_mean(x,axis=[1,2])
                    elif self.pooling == "max":
                        x=tf.math.reduce_max(x,axis=[1,2])
                return x
        else:
            x=self.layers(data,self.km)
            if self.include_top:
                x=tf.math.reduce_mean(x,axis=[1,2],keepdims=True)
                x=self.conv2d(x)
                x=tf.reshape(x,[x.shape[0],self.classes])
                x=self.head(x)
            else:
                if self.pooling == "avg":
                    x=tf.math.reduce_mean(x,axis=[1,2])
                elif self.pooling == "max":
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