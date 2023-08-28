import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_normalization import batch_normalization
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class _conv_block:
    def __init__(self, in_channels, filters, alpha, kernel=(3, 3), strides=[1, 1], dtype='float32'):
        filters = int(filters * alpha)
        self.conv2d=conv2d([kernel[0],kernel[1],in_channels,filters],strides=strides,padding='SAME',use_bias=False,dtype=dtype)
        self.batch_norm=batch_normalization(self.conv2d.output_size,keepdims=True,dtype=dtype)
        self.train_flag=True
        self.output_size=self.conv2d.output_size
        self.param=[self.conv2d.param,self.batch_norm.param]
    
    
    def output(self,data,train_flag=True):
        self.train_flag=train_flag
        x=self.conv2d.output(data)
        x=self.batch_norm.output(x,self.train_flag)
        return activation_dict['relu6'](x)


class _depthwise_conv_block:
    def __init__(self,in_channels,pointwise_conv_filters,alpha,depth_multiplier=1,strides=[1, 1],block_id=1,dtype='float32'):
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)
        self.strides=strides
        self.zeropadding2d=tf.pad
        self.depthwiseconv2d=depthwise_conv2d([3,3,in_channels,depth_multiplier],strides=[1,strides[0],strides[1],1],padding="SAME" if strides == [1, 1] else "VALID",use_bias=False,dtype=dtype)
        self.batch_norm1=batch_normalization(self.depthwiseconv2d.output_size,keepdims=True,dtype=dtype)
        self.conv2d=conv2d([1,1,self.depthwiseconv2d.output_size,pointwise_conv_filters],strides=[1, 1],padding='SAME',use_bias=False,dtype=dtype)
        self.batch_norm2=batch_normalization(self.conv2d.output_size,keepdims=True,dtype=dtype)
        self.train_flag=True
        self.output_size=self.conv2d.output_size
        self.param=[self.depthwiseconv2d.param,self.batch_norm1.param,self.conv2d.param,self.batch_norm2.param]
    
    
    def output(self,data,train_flag=True):
        self.train_flag=train_flag
        if self.strides == [1, 1]:
            x = data
        else:
            x = tf.pad(data, [[0, 0], [0, 1], [0, 1], [0, 0]])
        x=self.depthwiseconv2d.output(x)
        
        x=self.batch_norm1.output(x,self.train_flag)
        x=activation_dict['relu6'](x)
        x=self.conv2d.output(x)
        x=self.batch_norm2.output(x,self.train_flag)
        return activation_dict['relu6'](x)


class MobileNet:
    def __init__(self, in_channels=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, pooling=None, classes=1000):
        self.in_channels=in_channels
        self.alpha=alpha
        self.depth_multiplier=depth_multiplier
        self.dropout=dropout
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.optimizer=Adam()
        self.param=[]
        self.km=0
        
        
    def build(self,dtype='float32'):
        self.layers=Layers()
        self.layers.add(_conv_block(self.in_channels, 32, self.alpha, strides=[2, 2],dtype=dtype))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 64, self.alpha, self.depth_multiplier, block_id=1,dtype=dtype))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 128, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=2, dtype=dtype
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 128, self.alpha, self.depth_multiplier, block_id=3, dtype=dtype))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 256, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=4, dtype=dtype
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 256, self.alpha, self.depth_multiplier, block_id=5, dtype=dtype))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 512, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=6, dtype=dtype
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=7, dtype=dtype))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=8, dtype=dtype))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=9, dtype=dtype))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=10, dtype=dtype))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 512, self.alpha, self.depth_multiplier, block_id=11, dtype=dtype))
        self.layers.add(_depthwise_conv_block(
        self.layers.output_size, 1024, self.alpha, self.depth_multiplier, strides=[2, 2], block_id=12, dtype=dtype
        ))
        self.layers.add(_depthwise_conv_block(self.layers.output_size, 1024, self.alpha, self.depth_multiplier, block_id=13, dtype=dtype))
        self.conv2d=conv2d([1,1,self.layers.output_size,self.classes],padding='SAME',dtype=dtype)
        self.dense=dense([self.conv2d.output_size,self.classes],activation='softmax',dtype=dtype)
        self.bc=tf.Variable(0,dtype=dtype)
        self.param=[self.layers.param,self.dense.param]
        return
    
    
    def fp(self,data,p):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                x=self.layers.output(data)
                if self.include_top:
                    x=tf.math.reduce_mean(x,axis=[1,2],keepdims=True)
                    x=tf.nn.dropout(x,self.dropout)
                    x=self.conv2d.output(x)
                    x=tf.reshape(x,[x.shape[0],self.classes])
                    x=self.dense.output(x)
                else:
                    if self.pooling == "avg":
                        x=tf.math.reduce_mean(x,axis=[1,2])
                    elif self.pooling == "max":
                        x=tf.math.reduce_max(x,axis=[1,2])
        else:
            x=self.layers.output(data,self.km)
            if self.include_top:
                x=tf.math.reduce_mean(x,axis=[1,2],keepdims=True)
                x=self.conv2d.output(x)
                x=tf.reshape(x,[x.shape[0],self.classes])
                x=self.dense.output(x)
            else:
                if self.pooling == "avg":
                    x=tf.math.reduce_mean(x,axis=[1,2])
                elif self.pooling == "max":
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