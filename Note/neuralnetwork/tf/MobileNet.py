import tensorflow as tf
from Note import nn
from Note.nn.activation import activation_dict


class _conv_block:
    def __init__(self, in_channels, filters, alpha, kernel=(3, 3), strides=[1, 1], dtype='float32'):
        filters = int(filters * alpha)
        self.conv2d=nn.conv2d(filters,kernel,in_channels,strides=strides,padding='SAME',use_bias=False,dtype=dtype)
        self.batch_norm=nn.batch_norm(self.conv2d.output_size,dtype=dtype)
        self.train_flag=True
        self.output_size=self.conv2d.output_size
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        x=self.conv2d(data)
        x=self.batch_norm(x,self.train_flag)
        return activation_dict['relu6'](x)


class _depthwise_conv_block:
    def __init__(self,in_channels,pointwise_conv_filters,alpha,depth_multiplier=1,strides=[1, 1],block_id=1,dtype='float32'):
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)
        self.strides=strides
        self.zeropadding2d=nn.zeropadding2d()
        self.depthwiseconv2d=nn.depthwise_conv2d([3,3],depth_multiplier,in_channels,strides=[1,strides[0],strides[1],1],padding="SAME" if strides == [1, 1] else "VALID",use_bias=False,dtype=dtype)
        self.batch_norm1=nn.batch_norm(self.depthwiseconv2d.output_size,dtype=dtype)
        self.conv2d=nn.conv2d(pointwise_conv_filters,[1,1],self.depthwiseconv2d.output_size,strides=[1, 1],padding='SAME',use_bias=False,dtype=dtype)
        self.batch_norm2=nn.batch_norm(self.conv2d.output_size,dtype=dtype)
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


class MobileNet:
    def __init__(self, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, pooling=None, classes=1000):
        self.alpha=alpha
        self.depth_multiplier=depth_multiplier
        self.dropout=dropout
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.training=True
        
        
    def build(self,dtype='float32'):
        nn.Model.init()
        self.layers=nn.Layers()
        self.layers.add(_conv_block(3, 32, self.alpha, strides=[2, 2],dtype=dtype))
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
        self.conv2d=nn.conv2d(self.classes,[1,1],self.layers.output_size,padding='SAME',dtype=dtype)
        self.dense=nn.dense(self.classes,self.conv2d.output_size,activation='softmax',dtype=dtype)
        self.param=nn.Model.param
        return
    
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.dense_=self.nn.dense
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
    
    
    def __call__(self,data):
        x=self.layers(data,self.training)
        if self.include_top:
            x=tf.math.reduce_mean(x,axis=[1,2],keepdims=True)
            x=self.conv2d(x)
            x=tf.reshape(x,[x.shape[0],self.classes])
            x=self.dense(x)
        else:
            if self.pooling == "avg":
                x=tf.math.reduce_mean(x,axis=[1,2])
            elif self.pooling == "max":
                x=tf.math.reduce_max(x,axis=[1,2])
        return x