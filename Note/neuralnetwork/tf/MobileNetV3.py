import tensorflow as tf
from Note import nn
from Note.nn.layer.image_preprocessing.rescaling import rescaling
from Note.nn.activation import activation_dict


class MobileNetV3(nn.Model):
    def __init__(
        self,
        alpha=1.0,
        model_type="large",
        minimalistic=False,
        include_top=True,
        classes=1000,
        pooling=None,
        dropout_rate=0.2,
        include_preprocessing=True,
    ):
        super().__init__()
        if minimalistic:
            self.kernel = 3
            self.activation = relu
            self.se_ratio = None
        else:
            self.kernel = 5
            self.activation = hard_swish
            self.se_ratio = 0.25
        if model_type=='small':
            self.last_point_ch=1024
        else:
            self.last_point_ch=1280
        self.alpha=alpha
        self.model_type=model_type
        self.include_top=include_top
        self.classes=classes
        self.pooling=pooling
        self.dropout_rate=dropout_rate
        self.include_preprocessing=include_preprocessing
        
        if self.include_preprocessing:
            self.rescaling=rescaling(scale=1.0 / 127.5, offset=-1.0)
        self.layers=nn.Layers()
        self.layers.add(nn.conv2d(16,kernel_size=3,input_size=3,strides=(2, 2),padding="SAME",use_bias=False))
        self.layers.add(nn.batch_norm(epsilon=1e-3,momentum=0.999))
        self.layers.add(self.activation)
        
        if self.model_type=='small':
            self.layers.add(self.stack_fn_small(self.layers.output_size, self.kernel, self.activation, self.se_ratio))
        else:
            self.layers.add(self.stack_fn_large(self.layers.output_size, self.kernel, self.activation, self.se_ratio))
            
        last_conv_ch = _depth(self.layers.output_size * 6)
    
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if self.alpha > 1.0:
            last_point_ch = _depth(self.last_point_ch * self.alpha)
        else:
            last_point_ch = self.last_point_ch
        self.layers.add(nn.conv2d(last_conv_ch,kernel_size=1,padding="SAME",use_bias=False))
        self.layers.add(nn.batch_norm(epsilon=1e-3,momentum=0.999))
        self.layers.add(self.activation)
        if self.include_top:
            self.layers.add(nn.global_avg_pool2d(keepdims=True))
            self.layers.add(nn.conv2d(last_point_ch,kernel_size=1,padding="SAME",use_bias=True))
            self.layers.add(self.activation)
    
            if self.dropout_rate > 0:
                self.layers.add(nn.dropout(self.dropout_rate))
            
            self.head=self.conv2d(
                self.classes, self.layers.output_size, kernel_size=1, padding="SAME"
            )
        self.flatten=nn.flatten()
        
        self.training=True
    
    
    def stack_fn_small(self, in_channels, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * self.alpha)
        
        layers=nn.Layers()
    
        layers.add(_inverted_res_block(in_channels, 1, depth(16), 3, 2, se_ratio, relu, 0))
        layers.add(_inverted_res_block(layers.output_size, 72.0 / 16, depth(24), 3, 2, None, relu, 1))
        layers.add(_inverted_res_block(layers.output_size, 88.0 / 24, depth(24), 3, 1, None, relu, 2))
        layers.add(_inverted_res_block(layers.output_size, 4, depth(40), kernel, 2, se_ratio, activation, 3))
        layers.add(_inverted_res_block(layers.output_size, 6, depth(40), kernel, 1, se_ratio, activation, 4))
        layers.add(_inverted_res_block(layers.output_size, 6, depth(40), kernel, 1, se_ratio, activation, 5))
        layers.add(_inverted_res_block(layers.output_size, 3, depth(48), kernel, 1, se_ratio, activation, 6))
        layers.add(_inverted_res_block(layers.output_size, 3, depth(48), kernel, 1, se_ratio, activation, 7))
        layers.add(_inverted_res_block(layers.output_size, 6, depth(96), kernel, 2, se_ratio, activation, 8))
        layers.add(_inverted_res_block(layers.output_size, 6, depth(96), kernel, 1, se_ratio, activation, 9))
        layers.add(_inverted_res_block(
            layers.output_size, 6, depth(96), kernel, 1, se_ratio, activation, 10
        ))
        return layers
    
    
    def stack_fn_large(self, in_channels, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * self.alpha)
        
        layers=nn.Layers()
    
        layers.add(_inverted_res_block(in_channels, 1, depth(16), 3, 1, None, relu, 0))
        layers.add(_inverted_res_block(layers.output_size, 4, depth(24), 3, 2, None, relu, 1))
        layers.add(_inverted_res_block(layers.output_size, 3, depth(24), 3, 1, None, relu, 2))
        layers.add(_inverted_res_block(layers.output_size, 3, depth(40), kernel, 2, se_ratio, relu, 3))
        layers.add(_inverted_res_block(layers.output_size, 3, depth(40), kernel, 1, se_ratio, relu, 4))
        layers.add(_inverted_res_block(layers.output_size, 3, depth(40), kernel, 1, se_ratio, relu, 5))
        layers.add(_inverted_res_block(layers.output_size, 6, depth(80), 3, 2, None, activation, 6))
        layers.add(_inverted_res_block(layers.output_size, 2.5, depth(80), 3, 1, None, activation, 7))
        layers.add(_inverted_res_block(layers.output_size, 2.3, depth(80), 3, 1, None, activation, 8))
        layers.add(_inverted_res_block(layers.output_size, 2.3, depth(80), 3, 1, None, activation, 9))
        layers.add(_inverted_res_block(
            layers.output_size, 6, depth(112), 3, 1, se_ratio, activation, 10
        ))
        layers.add(_inverted_res_block(
            layers.output_size, 6, depth(112), 3, 1, se_ratio, activation, 11
        ))
        layers.add(_inverted_res_block(
            layers.output_size, 6, depth(160), kernel, 2, se_ratio, activation, 12
        ))
        layers.add(_inverted_res_block(
            layers.output_size, 6, depth(160), kernel, 1, se_ratio, activation, 13
        ))
        layers.add(_inverted_res_block(
            layers.output_size, 6, depth(160), kernel, 1, se_ratio, activation, 14
        ))
        return layers
    
    
    def __call__(self,data):
        if self.include_preprocessing:
            data=self.rescaling(data)
        x=self.layers(data,self.training)
        x=self.head(x)
        if self.include_top:
            x=self.flatten(x)
            x=self.activation(x)
        else:
            if self.pooling=="avg":
                x=tf.math.reduce_mean(x,axis=[1,2])
            elif self.pooling=="max":
                x=tf.math.reduce_max(x,axis=[1,2])
        return x


def relu(x):
    return activation_dict['relu'](x)


def hard_sigmoid(x):
    return activation_dict['relu6'](x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return tf.math.nn.multiply(x,hard_sigmoid(x))


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(in_channels, filters, se_ratio):
    layers=nn.Layers()
    layers.add(nn.identity(in_channels),save_data=True)
    layers.add(nn.global_avg_pool2d(keepdims=True))
    layers.add(nn.conv2d(_depth(filters * se_ratio),1,padding='SAME'))
    layers.add(activation_dict['relu'])
    layers.add(nn.conv2d(filters,1,padding='SAME'))
    layers.add(hard_sigmoid,save_data=True)
    layers.add(nn.multiply(),use_data=True)
    return layers


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
    def __init__(self, in_channels, expansion, filters, kernel_size, stride, se_ratio, activation, block_id):
        self.conv2d1=nn.conv2d(_depth(in_channels * expansion),1,in_channels,padding="SAME",use_bias=False)
        self.batch_normalization1=nn.batch_norm(self.conv2d1.output_size,epsilon=1e-3,momentum=0.999)
        if stride == 2:
            self.zeropadding2d=nn.zeropadding2d()
        self.depthwiseconv2d=nn.depthwise_conv2d(kernel_size,input_size=self.conv2d1.output_size,strides=stride,use_bias=False,padding="SAME" if stride == 1 else "VALID")
        self.batch_normalization2=nn.batch_norm(self.depthwiseconv2d.output_size,epsilon=1e-3,momentum=0.999)
        if se_ratio:
            self.layers=_se_block(self.depthwiseconv2d.output_size, _depth(in_channels * expansion), se_ratio)
            self.conv2d2=nn.conv2d(filters,1,self.layers.output_size,padding="SAME",use_bias=False)
        else:
            self.conv2d2=nn.conv2d(filters,1,self.depthwiseconv2d.output_size,padding="SAME",use_bias=False)
        self.batch_normalization3=nn.batch_norm(self.conv2d2.output_size,epsilon=1e-3,momentum=0.999)
        self.in_channels=in_channels
        self.filters=filters
        self.stride=stride
        self.se_ratio=se_ratio
        self.activation=activation
        self.block_id=block_id
        self.train_flag=True
        self.output_size=self.conv2d2.output_size
    
    
    def __call__(self,data,train_flag=True):
        self.train_flag=train_flag
        x=data
        shortcut=data
        if self.block_id:
            x=self.conv2d1(x)
            x=self.batch_normalization1(x,self.train_flag)
            x=self.activation(x)
        if self.stride==2:
            padding = correct_pad(x, 3)
            x=self.zeropadding2d(x, padding)
        x=self.depthwiseconv2d(x)
        x=self.batch_normalization2(x,self.train_flag)
        x=self.activation(x)
        if self.se_ratio:
            x=self.layers(x)
        x=self.conv2d2(x)
        x=self.batch_normalization3(x,self.train_flag)
        if self.stride == 1 and self.in_channels == self.filters:
            return shortcut+x
        return x