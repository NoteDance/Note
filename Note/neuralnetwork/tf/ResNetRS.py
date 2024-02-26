import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.avg_pool2d import avg_pool2d
from Note.nn.layer.global_avg_pool2d import global_avg_pool2d
from Note.nn.layer.batch_norm import batch_norm
from Note.nn.layer.identity import identity
from Note.nn.activation import activation_dict
from Note.nn.Layers import Layers
from typing import List
from typing import Dict
from Note.nn.Module import Module


def fixed_padding(inputs, kernel_size):
    """Pad the input along the spatial dimensions independently of input
    size."""
    pad_total = kernel_size[0] - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # Use ZeroPadding as to avoid TFOpLambda layer
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return padded_inputs


class Conv2DFixedPadding:
    def __init__(self,filters, kernel_size, strides, in_channels, dtype):
        """Conv2D block with fixed padding."""
        self.conv2d=conv2d(filters,kernel_size,in_channels,strides=[strides],padding='SAME' if strides==1 else 'VALID',use_bias=False,
                           weight_initializer=['VarianceScaling',2.0,'fan_out','truncated_normal'],dtype=dtype)
        self.kernel_size=kernel_size
        self.strides=strides
        self.output_size=filters
    
    
    def __call__(self,data):
        if self.strides > 1:
            data = fixed_padding(data, self.kernel_size)
        return self.conv2d(data)
    
    
def STEM(
    bn_momentum: float = 0.0,
    bn_epsilon: float = 1e-5,
    activation: str = "relu",
    dtype='float32'
):
    """ResNet-D type STEM block."""
    
    layers=Layers()
    
    # First stem block
    layers.add(Conv2DFixedPadding(
        filters=32, kernel_size=[3,3], strides=2, in_channels=3, dtype=dtype
    ))
    layers.add(batch_norm(
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        dtype=dtype
    ))
    layers.add(activation_dict[activation])

    # Second stem block
    layers.add(Conv2DFixedPadding(
        filters=32, kernel_size=[3,3], strides=1, in_channels=layers.output_size, dtype=dtype
    ))
    layers.add(batch_norm(
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        dtype=dtype
    ))
    layers.add(activation_dict[activation])

    # Final Stem block:
    layers.add(Conv2DFixedPadding(
        filters=64, kernel_size=[3,3], strides=1, in_channels=layers.output_size, dtype=dtype
    ))
    layers.add(batch_norm(
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        dtype=dtype
    ))
    layers.add(activation_dict[activation])

    # Replace stem max pool:
    layers.add(Conv2DFixedPadding(
        filters=64, kernel_size=[3,3], strides=2, in_channels=layers.output_size, dtype=dtype
    ))
    layers.add(batch_norm(
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        dtype=dtype
    ))
    layers.add(activation_dict[activation])
    return layers


def get_survival_probability(init_rate, block_num, total_blocks):
    """Get survival probability based on block number and initial rate."""
    return init_rate * float(block_num) / total_blocks


class SE:
    def __init__(self, in_filters: int, se_ratio: float = 0.25, expand_ratio: int = 1, in_channels=None, dtype='float32'):
        """Squeeze and Excitation block."""

        self.global_avg_pool2d=global_avg_pool2d()
        
        num_reduced_filters = max(1, int(in_filters * 4 * se_ratio))
        self.conv2d1=conv2d(num_reduced_filters,[1, 1],in_channels,strides=[1, 1],weight_initializer=['VarianceScaling',2.0,'fan_out','truncated_normal'],
                            padding="SAME",use_bias=True,activation="relu",dtype=dtype)
        self.conv2d2=conv2d(4*in_filters*expand_ratio,[1, 1],self.conv2d1.output_size,strides=[1, 1],
                            weight_initializer=['VarianceScaling',2.0,'fan_out','truncated_normal'],padding="SAME",
                            use_bias=True,activation="sigmoid",dtype=dtype)
        self.output_size=self.conv2d2.output_size
    
    
    def __call__(self,data):
        x=self.global_avg_pool2d(data)
        se_shape = (x.shape[0], 1, 1, x.shape[-1])
        x=tf.reshape(x,se_shape)
        x=self.conv2d1(x)
        x=self.conv2d2(x)
        return tf.math.multiply(data,x)


class BottleneckBlock:
    def __init__(
        self,
        in_channels,
        filters: int,
        strides: int,
        use_projection: bool,
        bn_momentum: float = 0.0,
        bn_epsilon: float = 1e-5,
        activation: str = "relu",
        se_ratio: float = 0.25,
        survival_probability: float = 0.8,
        dtype='float32'
        ):
        """Bottleneck block variant for residual networks with BN."""
    
        self.layers1=Layers()
        self.layers1.add(identity(in_channels))
    
        if use_projection:
            filters_out = filters * 4
            if strides == 2:
                self.layers1.add(avg_pool2d(
                    ksize=(2, 2),
                    strides=(2, 2),
                    padding="SAME",
                ))
                self.layers1.add(Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=[1,1],
                    strides=1,
                    in_channels=self.layers1.output_size,
                    dtype=dtype
                ))
            else:
                self.layers1.add(Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=[1,1],
                    strides=strides,
                    in_channels=self.layers1.output_size,
                    dtype=dtype
                ))
    
            self.layers1.add(batch_norm(
                momentum=bn_momentum,
                epsilon=bn_epsilon,
                dtype=dtype
            ))
        
        self.layers2=Layers()
        self.layers2.add(identity(in_channels))
    
        # First conv layer:
        self.layers2.add(Conv2DFixedPadding(
            filters=filters, kernel_size=[1,1], strides=1, in_channels=self.layers2.output_size, dtype=dtype
        ))
        self.layers2.add(batch_norm(
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            dtype=dtype
        ))
        self.layers2.add(activation_dict[activation])
    
        # Second conv layer:
        self.layers2.add(Conv2DFixedPadding(
            filters=filters,
            kernel_size=[3,3],
            strides=strides,
            in_channels=self.layers2.output_size,
            dtype=dtype
        ))
        self.layers2.add(batch_norm(
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            dtype=dtype
        ))
        self.layers2.add(activation_dict[activation])
    
        # Third conv layer:
        self.layers2.add(Conv2DFixedPadding(
            filters=filters * 4, kernel_size=[1,1], strides=1, in_channels=self.layers2.output_size, dtype=dtype
        ))
        self.layers2.add(batch_norm(
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            dtype=dtype
        ))
    
        if 0 < se_ratio < 1:
            self.layers2.add(SE(filters, se_ratio=se_ratio, in_channels=self.layers2.output_size))
        self.survival_probability=survival_probability
        self.activation=activation
        self.train_flag=True
        self.output_size=self.layers2.output_size
    
    
    def __call__(self,data,train_flag=True):
        shortcut=self.layers1(data,train_flag)
        x=self.layers2(data,train_flag)
        # Drop connect
        if train_flag==True and self.survival_probability:
            x = tf.nn.dropout(
                x,
                self.survival_probability,
                noise_shape=(None, 1, 1, 1),
            )
    
        x = x+shortcut
    
        return activation_dict[self.activation](x)


def BlockGroup(
    in_channels,
    filters,
    strides,
    num_repeats,
    se_ratio: float = 0.25,
    bn_epsilon: float = 1e-5,
    bn_momentum: float = 0.0,
    activation: str = "relu",
    survival_probability: float = 0.8,
    dtype='float32'
):
    """Create one group of blocks for the ResNet model."""
    
    layers=Layers()

    # Only the first block per block_group uses projection shortcut and
    # strides.
    layers.add(BottleneckBlock(
        in_channels,
        filters=filters,
        strides=strides,
        use_projection=True,
        se_ratio=se_ratio,
        bn_epsilon=bn_epsilon,
        bn_momentum=bn_momentum,
        activation=activation,
        survival_probability=survival_probability,
        dtype=dtype
    ))

    for i in range(1, num_repeats):
        in_channels=layers.output_size
        layers.add(BottleneckBlock(
            in_channels,   
            filters=filters,
            strides=1,
            use_projection=False,
            se_ratio=se_ratio,
            activation=activation,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
            survival_probability=survival_probability,
            dtype=dtype
        ))
    return layers


class ResNetRS:
    def __init__(
            self,
            bn_momentum=0.0,
            bn_epsilon=1e-5,
            activation: str = "relu",
            se_ratio=0.25,
            dropout_rate=0.25,
            drop_connect_rate=0.2,
            include_top=True,
            block_args: List[Dict[str, int]] = None,
            model_name="resnet-rs-50",
            pooling=None,
            classes=1000,
            include_preprocessing=True,
    ):
        self.depth=MODEL_DEPTH[model_name]
        self.bn_momentum=bn_momentum
        self.bn_epsilon=bn_epsilon
        self.activation=activation
        self.se_ratio=se_ratio
        self.block_args=block_args
        self.dropout_rate=dropout_rate
        self.drop_connect_rate=drop_connect_rate
        self.include_top=include_top
        self.classes=classes
        self.include_preprocessing=include_preprocessing
        self.training=True
    
        
    def build(self,dtype='float32'):
        Module.init()
        
        self.layers=Layers()
        # Build stem
        self.layers.add(STEM(bn_momentum=self.bn_momentum, bn_epsilon=self.bn_epsilon, activation=self.activation, dtype=dtype))
        
        # Build blocks
        if self.block_args is None:
            self.block_args = BLOCK_ARGS[self.depth]
        for i, args in enumerate(self.block_args):
            survival_probability = get_survival_probability(
                init_rate=self.drop_connect_rate,
                block_num=i + 2,
                total_blocks=len(self.block_args) + 1,
            )
    
            self.layers.add(BlockGroup(
                self.layers.output_size,
                filters=args["input_filters"],
                activation=self.activation,
                strides=(1 if i == 0 else 2),
                num_repeats=args["num_repeats"],
                se_ratio=self.se_ratio,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
                survival_probability=survival_probability,
                dtype=dtype
            ))
        self.dense=dense(self.classes,self.layers.output_size,activation='softmax',dtype=dtype)
        self.dtype=dtype
        self.param=Module.param
    
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
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
        if self.include_preprocessing:
            scale = tf.constant(1.0 / 255, dtype=self.dtype)
            rescaling_data = tf.multiply(data, scale)
            mean = tf.constant([0.485, 0.456, 0.406], dtype=self.dtype)
            variance = tf.constant([0.229**2, 0.224**2, 0.225**2], dtype=self.dtype)
            normalization_data = tf.nn.batch_norm(rescaling_data, mean, variance, None, None, 1e-12)
            data=normalization_data
        x=self.layers(data,self.training)
        # Build head:
        if self.include_top:
            x = tf.reduce_mean(x, axis=[1, 2])
            if self.dropout_rate > 0:
                x = tf.nn.dropout(x,self.dropout_rate)
            x=self.dense(x)
        else:
            if self.pooling == "avg":
                x = tf.reduce_mean(x, axis=[1, 2])
            elif self.pooling == "max":
                x = tf.reduce_max(x, axis=[1, 2])
        return x


BLOCK_ARGS = {
    50: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 6},
        {"input_filters": 512, "num_repeats": 3},
    ],
    101: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 23},
        {"input_filters": 512, "num_repeats": 3},
    ],
    152: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 8},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    200: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 24},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    270: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 29},
        {"input_filters": 256, "num_repeats": 53},
        {"input_filters": 512, "num_repeats": 4},
    ],
    350: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 36},
        {"input_filters": 256, "num_repeats": 72},
        {"input_filters": 512, "num_repeats": 4},
    ],
    420: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 44},
        {"input_filters": 256, "num_repeats": 87},
        {"input_filters": 512, "num_repeats": 4},
    ],
}


MODEL_DEPTH = {
    "resnet-rs-50": 50,

    "resnet-rs-101": 101,
    
    "resnet-rs-152": 152,

    "resnet-rs-200": 200,

    "resnet-rs-270": 270,

    "resnet-rs-350": 350,

    "resnet-rs-420": 420,   
}