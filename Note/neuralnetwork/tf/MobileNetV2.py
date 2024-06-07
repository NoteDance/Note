import tensorflow as tf
from Note import nn
from Note.nn.activation import activation_dict


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
    def __init__(self, in_channels=None, expansion=None, stride=None, alpha=None, filters=None, block_id=None):
        self.conv2d1=nn.conv2d(expansion * in_channels,[1,1],in_channels,padding="SAME",use_bias=False)
        self.batch_normalization1=nn.batch_norm(self.conv2d1.output_size,momentum=0.999)
        self.zeropadding2d=nn.zeropadding2d()
        self.depthwiseconv2d=nn.depthwise_conv2d([3,3],1,self.conv2d1.output_size,strides=[1,stride,stride,1],use_bias=False,padding="SAME" if stride == 1 else "VALID")
        self.batch_normalization2=nn.batch_norm(self.depthwiseconv2d.output_size,momentum=0.999)
        pointwise_conv_filters = int(filters * alpha)
        # Ensure the number of filters on the last 1x1 convolution is divisible by
        # 8.
        self.pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        self.conv2d2=nn.conv2d(self.pointwise_filters,[1,1],self.depthwiseconv2d.output_size,padding="SAME",use_bias=False)
        self.batch_normalization3=nn.batch_norm(self.conv2d2.output_size,momentum=0.999)
        self.in_channels=in_channels
        self.stride=stride
        self.block_id=block_id
        self.train_flag=True
        self.output_size=self.conv2d2.output_size
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        x=data
        if self.block_id:
            x=self.conv2d1(x)
            x=self.batch_normalization1(x,self.train_flag)
            x=activation_dict['relu6'](x)
        if self.stride==2:
            padding = correct_pad(x, 3)
            x=self.zeropadding2d(x, padding)
        x=self.depthwiseconv2d(x)
        x=self.batch_normalization2(x,self.train_flag)
        x=activation_dict['relu6'](x)
        x=self.conv2d2(x)
        x=self.batch_normalization3(x,self.train_flag)
        if self.in_channels == self.pointwise_filters and self.stride == 1:
            return data+x
        return x


class MobileNetV2(nn.Model):
    def __init__(self,alpha=1.0,classes=1000,include_top=True,pooling=None):
        super().__init__()
        self.classes=classes
        self.alpha=alpha
        self.include_top=include_top
        self.pooling=pooling
        self.first_block_filters = _make_divisible(32 * alpha, 8)
        
        self.layers=nn.Layers()
        
        self.layers.add(nn.conv2d(self.first_block_filters,[3,3],3,strides=[2,2],padding='SAME',use_bias=False))
        self.layers.add(nn.batch_norm(momentum=0.999))
        
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=16, alpha=self.alpha, stride=1, expansion=1, block_id=0))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=24, alpha=self.alpha, stride=2, expansion=6, block_id=1))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=24, alpha=self.alpha, stride=1, expansion=6, block_id=2))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=32, alpha=self.alpha, stride=2, expansion=6, block_id=3))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=4))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=5))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=2, expansion=6, block_id=6))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=7))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=8))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=9))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=10))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=11))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=12))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=160, alpha=self.alpha, stride=2, expansion=6, block_id=13))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=14))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=15))
        self.layers.add(_inverted_res_block(self.layers.output_size, filters=320, alpha=self.alpha, stride=1, expansion=6, block_id=16))

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we increase the number of output
        # channels.
        if self.alpha > 1.0:
            last_block_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280
            
        self.layers.add(nn.conv2d(last_block_filters,[1,1],use_bias=False))
        self.layers.add(nn.batch_norm(momentum=0.999))
        
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