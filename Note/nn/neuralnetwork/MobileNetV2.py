import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_normalization import batch_normalization
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple.
    """
    input_size = inputs.shape[1:3]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


class _inverted_res_block:
    def __init__(self, in_channels=None, expansion=None, stride=None, alpha=None, filters=None, block_id=None, dtype='float32'):
        self.conv2d1=conv2d([1,1,in_channels,expansion * in_channels],padding="SAME",use_bias=False,dtype=dtype)
        self.batch_normalization1=batch_normalization(self.conv2d1.output_size,momentum=0.999,dtype=dtype,keepdims=True)
        self.zeropadding2d=tf.pad
        self.depthwiseconv2d=depthwise_conv2d([3,3,self.conv2d1.output_size,1],strides=[1,stride,stride,1],use_bias=False,padding="SAME" if stride == 1 else "VALID",dtype=dtype)
        self.batch_normalization2=batch_normalization(self.depthwiseconv2d.output_size,momentum=0.999,dtype=dtype,keepdims=True)
        pointwise_conv_filters = int(filters * alpha)
        # Ensure the number of filters on the last 1x1 convolution is divisible by
        # 8.
        self.pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        self.conv2d2=conv2d([1,1,self.depthwiseconv2d.output_size,self.pointwise_filters],padding="SAME",use_bias=False,dtype=dtype)
        self.batch_normalization3=batch_normalization(self.conv2d2.output_size,momentum=0.999,dtype=dtype,keepdims=True)
        self.in_channels=in_channels
        self.stride=stride
        self.block_id=block_id
        self.train_flag=True
        self.output_size=self.conv2d2.output_size
        self.param=[self.conv2d1.param,self.batch_normalization1.param,self.depthwiseconv2d.param,self.batch_normalization2.param,
                    self.conv2d2.param,self.batch_normalization3.param]
    
    
    def output(self,data,train_flag=True):
        self.train_flag=train_flag
        x=data
        if self.block_id:
            x=self.conv2d1.output(x)
            if self.train_flag:
                x=self.batch_normalization1.output(x)
            x=activation_dict['relu6'](x)
        if self.stride==2:
            padding = correct_pad(x, 3)
            x=self.zeropadding2d(x, [[0, 0], padding[0], padding[1], [0, 0]])
        x=self.depthwiseconv2d.output(x)
        if self.train_flag:
            x=self.batch_normalization2.output(x)
        x=activation_dict['relu6'](x)
        x=self.conv2d2.output(x)
        if self.train_flag:
            x=self.batch_normalization3.output(x)
        if self.in_channels == self.pointwise_filters and self.stride == 1:
            return data+x
        return x


class MobileNetV2:
    def __init__(self,in_channels,classes=1000,alpha=1.0,include_top=True,pooling=None):
        self.in_channels=in_channels
        self.classes=classes
        self.alpha=alpha
        self.include_top=include_top
        self.pooling=pooling
        self.first_block_filters = _make_divisible(32 * alpha, 8)
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.optimizer=Adam()
        self.param=[]
        self.km=0
        
        
    def build(self,dtype='float32'):
        self.conv2d1=conv2d([3,3,self.in_channels,self.first_block_filters],strides=[2,2],padding='SAME',use_bias=False,dtype=dtype)
        self.batch_normalization1=batch_normalization(self.conv2d1.output_size,momentum=0.999,keepdims=True)
        
        self.layers=Layers()
        self.layers.add(_inverted_res_block(self.conv2d1.output_size, filters=16, alpha=self.alpha, stride=1, expansion=1, block_id=0, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=24, alpha=self.alpha, stride=2, expansion=6, block_id=1, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=24, alpha=self.alpha, stride=1, expansion=6, block_id=2, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=32, alpha=self.alpha, stride=2, expansion=6, block_id=3, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=4, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=5, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=2, expansion=6, block_id=6, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=7, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=8, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=9, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=10, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=11, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=12, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=160, alpha=self.alpha, stride=2, expansion=6, block_id=13, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=14, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=15, dtype=dtype))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=320, alpha=self.alpha, stride=1, expansion=6, block_id=16, dtype=dtype))

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we increase the number of output
        # channels.
        if self.alpha > 1.0:
            last_block_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280
            
        self.conv2d2=conv2d([1,1,self.layers.output_size,last_block_filters],use_bias=False,dtype=dtype)
        self.batch_normalization2=batch_normalization(self.conv2d2.output_size,momentum=0.999,keepdims=True)
        
        self.dense=dense([self.conv2d2.output_size,self.classes],activation='softmax',dtype=dtype)
        self.bc=tf.Variable(0,dtype=dtype)
        self.param=[self.conv2d1.param,self.batch_normalization1.param,self.layers.param,self.conv2d2.param,
                    self.batch_normalization2.param,self.dense.param]
    
    
    def fp(self,data,p):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                x=self.conv2d1.output(data)
                x=self.batch_normalization1.output(x)
                
                x=self.layers.output(x)
                
                x=self.conv2d2.output(x)
                x=self.batch_normalization2.output(x)

        else:
            x=self.conv2d1.output(data)
            x=self.batch_normalization1.output(x)
            
            x=self.layers.output(x,self.km)
            
            x=self.conv2d2.output(x)

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