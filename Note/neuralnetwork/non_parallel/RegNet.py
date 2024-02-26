import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.global_avg_pool2d import global_avg_pool2d
from Note.nn.layer.global_max_pool2d import global_max_pool2d
from Note.nn.layer.batch_norm import batch_norm
from Note.nn.layer.identity import identity
from Note.nn.layer.add import add
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
from Note.nn.Module import Module


def PreStem(x,dtype='float32'):
    x = tf.cast(x, dtype)
    x = tf.math.divide(x, tf.cast(255.0,dtype))
    return x


def Stem(in_channels,dtype='float32'):
    layers=Layers()
    layers.add(conv2d(32,[3,3],in_channels,strides=[2],use_bias=False,padding='SAME',weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    layers.add(activation_dict['relu'])
    return layers


def SqueezeAndExciteBlock(in_channels, filters_in, se_filters, dtype='float32'):
    layers=Layers()
    layers.add(identity(in_channels),save_data=True)
    layers.add(global_avg_pool2d(keepdims=True))
    layers.add(conv2d(se_filters,[1,1],activation='relu',weight_initializer='He',dtype=dtype))
    layers.add(conv2d(filters_in,[1,1],activation='sigmoid',weight_initializer='He',dtype=dtype))
    layers.add(tf.math.multiply,use_data=True)
    return layers


def XBlock(in_channels, filters_in, filters_out, group_width, stride=1, dtype='float32'):
    if filters_in != filters_out and stride == 1:
        raise ValueError(
            f"Input filters({filters_in}) and output "
            f"filters({filters_out}) "
            f"are not equal for stride {stride}. Input and output filters "
            f"must be equal for stride={stride}."
        )

    # Declare layers
    groups = filters_out // group_width
    
    layers=Layers()
    if stride!=1:
        layers.add(conv2d(filters_out,[1,1],in_channels,strides=[2],use_bias=False,weight_initializer='He',dtype=dtype))
        layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    else:
        layers.add(identity(in_channels),save_data=True)

    # Build block
    # conv_1x1_1
    layers.add(conv2d(filters_out,[1,1],use_bias=False,weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    layers.add(activation_dict['relu'])

    # conv_3x3
    layers.add(conv2d(filters_out,[3,3],layers.output_size//groups,use_bias=False,strides=[stride],padding='SAME',weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    layers.add(activation_dict['relu'])

    # conv_1x1_2
    layers.add(conv2d(filters_out,[1,1],use_bias=False,weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype),save_data=True)
    
    layers.add(add(),use_data=True)
    
    layers.add(activation_dict['relu'])

    return layers


def YBlock(
    in_channels,
    filters_in,
    filters_out,
    group_width,
    stride=1,
    squeeze_excite_ratio=0.25,
    dtype='float32'
):
    if filters_in != filters_out and stride == 1:
        raise ValueError(
            f"Input filters({filters_in}) and output "
            f"filters({filters_out}) "
            f"are not equal for stride {stride}. Input and output filters "
            f"must be equal for stride={stride}."
        )

    groups = filters_out // group_width
    se_filters = int(filters_in * squeeze_excite_ratio)

    layers=Layers()
    if stride!=1:
        layers.add(conv2d(filters_out,[1,1],in_channels,strides=[2],use_bias=False,weight_initializer='He',dtype=dtype))
        layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    else:
        layers.add(identity(in_channels),save_data=True)

    # Build block
    # conv_1x1_1
    layers.add(conv2d(filters_out,[1,1],use_bias=False,weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    layers.add(activation_dict['relu'])

    # conv_3x3
    layers.add(conv2d(filters_out,[3,3],layers.output_size//groups,use_bias=False,strides=[stride],padding='SAME',weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    layers.add(activation_dict['relu'])

    # Squeeze-Excitation block
    layers.add(SqueezeAndExciteBlock(layers.output_size, filters_out, se_filters, dtype=dtype))

    # conv_1x1_2
    layers.add(conv2d(filters_out,[1,1],use_bias=False,weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype),save_data=True)
    
    layers.add(add(),use_data=True)

    layers.add(activation_dict['relu'])

    return layers


def ZBlock(
    in_channels,
    filters_in,
    filters_out,
    group_width,
    stride=1,
    squeeze_excite_ratio=0.25,
    bottleneck_ratio=0.25,
    dtype='float32'
):
    if filters_in != filters_out and stride == 1:
        raise ValueError(
            f"Input filters({filters_in}) and output filters({filters_out})"
            f"are not equal for stride {stride}. Input and output filters "
            f"must be equal for stride={stride}."
        )

    groups = filters_out // group_width
    se_filters = int(filters_in * squeeze_excite_ratio)

    inv_btlneck_filters = int(filters_out / bottleneck_ratio)
    
    layers=Layers()
    layers.add(identity(in_channels),save_data=True)
    
    # Build block
    # conv_1x1_1
    layers.add(conv2d(inv_btlneck_filters,[1,1],use_bias=False,weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(layers.output_size,momentum=0.9,epsilon=1e-5,dtype=dtype))
    layers.add(activation_dict['silu'])

    # conv_3x3
    layers.add(conv2d(inv_btlneck_filters,[3,3],layers.output_size//groups,use_bias=False,strides=[stride],padding='SAME',weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype))
    layers.add(activation_dict['silu'])

    # Squeeze-Excitation block
    layers.add(SqueezeAndExciteBlock(layers.output_size, inv_btlneck_filters, se_filters, dtype=dtype))

    # conv_1x1_2
    layers.add(conv2d(filters_out,[1,1],layers.output_size,use_bias=False,weight_initializer='He',dtype=dtype))
    layers.add(batch_norm(momentum=0.9,epsilon=1e-5,dtype=dtype),save_data=True)
    
    if stride == 1:
        layers.add(tf.math.add,use_data=True)
    return layers


def Stage(in_channels, block_type, depth, group_width, filters_in, filters_out, dtype='float32'):
    layers=Layers()
    if block_type == "X":
        layers.add(XBlock(
            in_channels,
            filters_in,
            filters_out,
            group_width,
            stride=2,
        ))
        for i in range(1, depth):
            layers.add(XBlock(
                layers.output_size,
                filters_out,
                filters_out,
                group_width,
            ))
    elif block_type == "Y":
        layers.add(YBlock(
            in_channels,
            filters_in,
            filters_out,
            group_width,
            stride=2,
            ))
        for i in range(1, depth):
            layers.add(YBlock(
                layers.output_size,
                filters_out,
                filters_out,
                group_width,
            ))
    elif block_type == "Z":
        layers.add(ZBlock(
            in_channels,
            filters_in,
            filters_out,
            group_width,
            stride=2,
            ))
        for i in range(1, depth):
            layers.add(ZBlock(
                layers.output_size,
                filters_out,
                filters_out,
                group_width,
                ))
    else:
        raise NotImplementedError(
            f"Block type `{block_type}` not recognized."
            "block_type must be one of (`X`, `Y`, `Z`). "
        )
    return layers


class RegNet:
    def __init__(self,
    model_name='x002',
    include_preprocessing=True,
    include_top=True,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    ):
        self.depths=MODEL_CONFIGS[model_name]['depths']
        self.widths=MODEL_CONFIGS[model_name]['widths']
        self.group_width=MODEL_CONFIGS[model_name]['group_width']
        self.block_type=MODEL_CONFIGS[model_name]['block_type']
        self.include_preprocessing=include_preprocessing
        self.include_top=include_top
        self.pooling=pooling
        self.classes=classes
        self.classifier_activation=classifier_activation
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.km=0
        
    
    def build(self,dtype='float32'):
        Module.init()
        self.dtype=dtype
        self.layers=Layers()
        self.layers.add(Stem(3,dtype))
        in_channels = 32
        for num_stage in range(4):
            depth = self.depths[num_stage]
            out_channels = self.widths[num_stage]
            
            self.layers.add(Stage(
                self.layers.output_size,
                self.block_type,
                depth,
                self.group_width,
                in_channels,
                out_channels,
            ))
            in_channels = out_channels
        if self.include_top:
            self.global_avg_pool2d=global_avg_pool2d()
            self.dense=dense(self.classes,self.layers.output_size,activation=self.classifier_activation,dtype=dtype)
        else:
            if self.pooling == "avg":
                self.global_avg_pool2d=global_avg_pool2d()
            elif self.pooling == "max":
                self.global_max_pool2d=global_max_pool2d()
        self.opt=tf.keras.optimizers.Adam()
        self.param=Module.param
    
    
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.dense_=self.dense
            self.dense=dense(classes,self.dense.input_size,activation=self.classifier_activation,dtype=self.dense.dtype)
            param.extend(self.dense.param)
            self.param=param
            self.opt.lr=lr
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
    
    
    def fp(self,data):
        x = data
        if self.include_preprocessing:
            x = PreStem(x,self.dtype)
        x = self.layers(x,self.km)
        if self.include_top:
            x = self.global_avg_pool2d(x)
            x = self.dense(x)
        else:
            if self.pooling=="avg":
                x=tf.math.reduce_mean(x,axis=[1,2])
            elif self.pooling=="max":
                x=tf.math.reduce_max(x,axis=[1,2])
        return x
        
    
    def loss(self,output,labels):
        loss=self.loss_object(labels,output)
        return loss


MODEL_CONFIGS = {
    "x002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "block_type": "X",
    },
    "x004": {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "group_width": 16,
        "block_type": "X",
    },
    "x006": {
        "depths": [1, 3, 5, 7],
        "widths": [48, 96, 240, 528],
        "group_width": 24,
        "block_type": "X",
    },
    "x008": {
        "depths": [1, 3, 7, 5],
        "widths": [64, 128, 288, 672],
        "group_width": 16,
        "block_type": "X",
    },
    "x016": {
        "depths": [2, 4, 10, 2],
        "widths": [72, 168, 408, 912],
        "group_width": 24,
        "block_type": "X",
    },
    "x032": {
        "depths": [2, 6, 15, 2],
        "widths": [96, 192, 432, 1008],
        "group_width": 48,
        "block_type": "X",
    },
    "x040": {
        "depths": [2, 5, 14, 2],
        "widths": [80, 240, 560, 1360],
        "group_width": 40,
        "block_type": "X",
    },
    "x064": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 392, 784, 1624],
        "group_width": 56,
        "block_type": "X",
    },
    "x080": {
        "depths": [2, 5, 15, 1],
        "widths": [80, 240, 720, 1920],
        "group_width": 120,
        "block_type": "X",
    },
    "x120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "block_type": "X",
    },
    "x160": {
        "depths": [2, 6, 13, 1],
        "widths": [256, 512, 896, 2048],
        "group_width": 128,
        "block_type": "X",
    },
    "x320": {
        "depths": [2, 7, 13, 1],
        "widths": [336, 672, 1344, 2520],
        "group_width": 168,
        "block_type": "X",
    },
    "y002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "block_type": "Y",
    },
    "y004": {
        "depths": [1, 3, 6, 6],
        "widths": [48, 104, 208, 440],
        "group_width": 8,
        "block_type": "Y",
    },
    "y006": {
        "depths": [1, 3, 7, 4],
        "widths": [48, 112, 256, 608],
        "group_width": 16,
        "block_type": "Y",
    },
    "y008": {
        "depths": [1, 3, 8, 2],
        "widths": [64, 128, 320, 768],
        "group_width": 16,
        "block_type": "Y",
    },
    "y016": {
        "depths": [2, 6, 17, 2],
        "widths": [48, 120, 336, 888],
        "group_width": 24,
        "block_type": "Y",
    },
    "y032": {
        "depths": [2, 5, 13, 1],
        "widths": [72, 216, 576, 1512],
        "group_width": 24,
        "block_type": "Y",
    },
    "y040": {
        "depths": [2, 6, 12, 2],
        "widths": [128, 192, 512, 1088],
        "group_width": 64,
        "block_type": "Y",
    },
    "y064": {
        "depths": [2, 7, 14, 2],
        "widths": [144, 288, 576, 1296],
        "group_width": 72,
        "block_type": "Y",
    },
    "y080": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 448, 896, 2016],
        "group_width": 56,
        "block_type": "Y",
    },
    "y120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "block_type": "Y",
    },
    "y160": {
        "depths": [2, 4, 11, 1],
        "widths": [224, 448, 1232, 3024],
        "group_width": 112,
        "block_type": "Y",
    },
    "y320": {
        "depths": [2, 5, 12, 1],
        "widths": [232, 696, 1392, 3712],
        "group_width": 232,
        "block_type": "Y",
    },
}