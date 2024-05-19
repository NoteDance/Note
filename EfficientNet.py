import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_norm import batch_norm_
from Note.nn.layer.dropout import dropout
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.global_avg_pool2d import global_avg_pool2d
from Note.nn.layer.global_max_pool2d import global_max_pool2d
from Note.nn.layer.norm import norm
from Note.nn.layer.image_preprocessing.rescaling import rescaling
from Note.nn.layer.reshape import reshape
from Note.nn.layer.identity import identity
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
import copy
from Note.nn.Model import Model


class EfficientNet:
    """Instantiates the EfficientNet architecture.
    
    Args:
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False.
          It should have exactly 3 inputs channels.
      model_name: string, model name.
      default_size: integer, default input image size.
      dropout_rate: float, dropout rate before final classifier layer.
      drop_connect_rate: float, dropout rate at skip connections.
      depth_divisor: integer, a unit of network width.
      activation: activation function.
      blocks_args: list of dicts, parameters to construct block modules.
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.

      pooling: optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.
    
    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """
    def __init__(
                self,
                input_shape,
                model_name='B0',
                drop_connect_rate=0.2,
                depth_divisor=8,
                activation="swish",
                blocks_args="default",
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax",
                dtype='float32'
            ):
        if weights == "imagenet":
            # Note that the normaliztion layer uses square value of STDDEV as the
            # variance for the layer: result = (input - mean) / sqrt(var)
            # However, the original implemenetation uses (input - mean) / var to
            # normalize the input, we need to divide another sqrt(var) to match the
            # original implementation.
            # See https://github.com/tensorflow/tensorflow/issues/49930 for more
            # details
            self.rescaling=rescaling([1.0 / tf.math.sqrt(stddev) for stddev in IMAGENET_STDDEV_RGB])
        else:
            self.rescaling=rescaling(1.0 / 255.0)
        if blocks_args == "default":
            self.blocks_args = DEFAULT_BLOCKS_ARGS
    
        if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
            raise ValueError(
                "The `weights` argument should be either "
                "`None` (random initialization), `imagenet` "
                "(pre-training on ImageNet), "
                "or the path to the weights file to be loaded."
            )
    
        if weights == "imagenet" and include_top and classes != 1000:
            raise ValueError(
                'If using `weights` as `"imagenet"` with `include_top`'
                " as true, `classes` should be 1000"
            )
        self.width_coefficient=MODEL_CONFIG[model_name]['width_coefficient']
        self.depth_coefficient=MODEL_CONFIG[model_name]['depth_coefficient']
        self.dropout_rate=MODEL_CONFIG[model_name]['dropout_rate']
        self.drop_connect_rate=drop_connect_rate
        self.depth_divisor=depth_divisor
        self.activation=activation
        self.classes=classes # store the number of classes
        self.include_top=include_top
        self.pooling=pooling
        self.classifier_activation=classifier_activation
        self.dtype=dtype
        self.norm=norm(input_shape,dtype=dtype)
        self.training=True
    
    
    def build(self):
        Model.init()
        
        def round_filters(filters, divisor=self.depth_divisor):
            """Round number of filters based on depth multiplier."""
            filters *= self.width_coefficient
            new_filters = max(
                divisor, int(filters + divisor / 2) // divisor * divisor
            )
            # Make sure that round down does not go down by more than 10%.
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)
    
        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(tf.math.ceil(self.depth_coefficient * repeats))
    
        # Build stem
        self.zeropadding2d=zeropadding2d()
        self.layers1=Layers()
        self.layers1.add(conv2d(round_filters(32),[3,3],3,strides=2,padding="VALID",use_bias=False,
                           weight_initializer=CONV_KERNEL_INITIALIZER,dtype=self.dtype))
        self.layers1.add(batch_norm_(axis=-1,dtype=self.dtype))
        self.layers1.add(activation_dict[self.activation])    
    
        # Build blocks
        blocks_args = copy.deepcopy(self.blocks_args)
        
        b = 0
        blocks = float(sum(round_repeats(args["repeats"]) for args in blocks_args))
        self.layers2=Layers()
        for i, args in enumerate(blocks_args):
            assert args["repeats"] > 0
            # Update block input and output filters based on depth multiplier.
            args["filters_in"] = round_filters(args["filters_in"])
            args["filters_out"] = round_filters(args["filters_out"])
            if i==0:
                in_channels=self.layers1.output_size
            for j in range(round_repeats(args.pop("repeats"))):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]
                self.layers2.add(block(
                    in_channels,
                    self.activation,
                    self.drop_connect_rate * b / blocks,
                    **args,
                    dtype=self.dtype
                ))
                b += 1
                in_channels=self.layers2.output_size
        
        # Build top
        self.layers3=Layers()
        self.layers3.add(conv2d(
            round_filters(1280),
            [1,1],
            self.layers2.output_size,
            padding="SAME",
            use_bias=False,
            weight_initializer=CONV_KERNEL_INITIALIZER,
            dtype=self.dtype
        ))
        self.layers3.add(batch_norm_(axis=-1,dtype=self.dtype))
        self.layers3.add(activation_dict[self.activation])
        if self.include_top:
            self.global_avg_pool2d=global_avg_pool2d()
            if self.dropout_rate > 0:
                self.dropout=dropout(self.dropout_rate)
            self.dense=dense(self.classes,self.layers3.output_size,activation=self.classifier_activation,weight_initializer=DENSE_KERNEL_INITIALIZER,dtype=self.dtype)
        else:
            if self.pooling == "avg":
                self.global_avg_pool2d=global_avg_pool2d()
            elif self.pooling == "max":
                self.global_max_pool2d=global_max_pool2d()
        self.param=Model.param
    
    
    def fine_tuning(self,classes=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param.copy()
            self.dense_=self.dense
            self.dense=dense(classes,self.dense.input_size,activation=self.classifier_activation,dtype=self.dense.dtype)
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
        data=self.rescaling(data)
        data=self.norm(data)
        x=self.zeropadding2d(data,correct_pad(data,3))
        x=self.layers1(x,self.training)
        x=self.layers2(x,self.training)
        x=self.layers3(x,self.training)
        if self.include_top:
            x=self.global_avg_pool2d(x)
            if self.dropout_rate > 0:
                x=self.dropout(x)
            x=self.dense(x)
        else:
            if self.pooling == "avg":
                x=self.global_avg_pool2d(x)
            else:
                x=self.global_max_pool2d(x)
        return x


class block:
    def __init__(self,
        in_channels,
        activation="swish",
        drop_rate=0.0,
        filters_in=32,
        filters_out=16,
        kernel_size=3,
        strides=1,
        expand_ratio=1,
        se_ratio=0.0,
        id_skip=True,
        dtype='float32'
        ):
        """An inverted residual block.
    
        Args:
            in_channels: input channels.
            activation: activation function.
            drop_rate: float between 0 and 1, fraction of the input units to drop.
            filters_in: integer, the number of input filters.
            filters_out: integer, the number of output filters.
            kernel_size: integer, the dimension of the convolution window.
            strides: integer, the stride of the convolution.
            expand_ratio: integer, scaling coefficient for the input filters.
            se_ratio: float between 0 and 1, fraction to squeeze the input filters.
            id_skip: boolean.
    
        Returns:
            output tensor for the block.
        """
        self.strides=strides
        self.se_ratio=se_ratio
        self.id_skip=id_skip
        self.filters_in=filters_in
        self.filters_out=filters_out
        self.kernel_size=kernel_size
        self.drop_rate=drop_rate
        
        self.layers1=Layers()
        # Expansion phase
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            self.layers1.add(conv2d(filters,[1,1],
                              in_channels,padding="SAME",
                              use_bias=False,
                              weight_initializer=CONV_KERNEL_INITIALIZER,dtype=dtype))
            self.layers1.add(batch_norm_(axis=-1,dtype=dtype))
            self.layers1.add(activation_dict[activation])
        else:
            self.layers1.add(identity(in_channels))

        # Depthwise Convolution
        if strides == 2:
            self.zeropadding2d=zeropadding2d()
            conv_pad = "VALID"
        else:
            conv_pad = "SAME"
        self.layers2=Layers()
        self.layers2.add(depthwise_conv2d(
                                            [kernel_size,kernel_size],
                                            input_size=self.layers1.output_size,
                                            strides=strides,
                                            padding=conv_pad,
                                            use_bias=False,
                                            weight_initializer=CONV_KERNEL_INITIALIZER,
                                            dtype=dtype))
        self.layers2.add(batch_norm_(axis=-1,dtype=dtype))
        self.layers2.add(activation_dict[activation])

        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            self.global_avg_pool2d=global_avg_pool2d()
            self.reshape=reshape((1, 1, filters))
            self.conv2d1=conv2d(filters_se,[1,1],self.layers2.output_size,padding="SAME",
                               activation=activation,weight_initializer=CONV_KERNEL_INITIALIZER,dtype=dtype)
            self.conv2d2=conv2d(filters,[1,1],padding="SAME",activation="sigmoid",
                                weight_initializer=CONV_KERNEL_INITIALIZER,dtype=dtype)

        # Output phase
        self.layers3=Layers()
        self.layers3.add(conv2d(filters_out,[1,1],self.conv2d2.output_size,padding="SAME",
                            use_bias=False,weight_initializer=CONV_KERNEL_INITIALIZER,dtype=dtype))
        self.layers3.add(batch_norm_(axis=-1,dtype=dtype))
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                self.dropout=dropout(drop_rate,noise_shape=(None, 1, 1, 1))
        self.output_size=self.layers3.output_size
        self.train_flag=True
    
    
    def __call__(self,data,train_flag=True):
        x=self.layers1(data,train_flag)
        if self.strides==2:
            x=self.zeropadding2d(x,correct_pad(x,self.kernel_size))
        x=self.layers2(x,train_flag)
        if 0 < self.se_ratio <= 1:
            se=self.global_avg_pool2d(x)
            se=self.reshape(se)
            se=self.conv2d1(se)
            se=self.conv2d2(se)
            x=tf.math.multiply(x,se)
        x=self.layers3(x,train_flag)
        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out:
            if self.drop_rate > 0:
                x=self.dropout(x,train_flag)
            x=tf.math.add(x,data)
        return x


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


DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]


MODEL_CONFIG={
    'B0':{'width_coefficient':1.0,'depth_coefficient':1.0,'dropout_rate':0.2},
    'B1':{'width_coefficient':1.0,'depth_coefficient':1.1,'dropout_rate':0.2},
    'B2':{'width_coefficient':1.1,'depth_coefficient':1.2,'dropout_rate':0.3},
    'B3':{'width_coefficient':1.2,'depth_coefficient':1.4,'dropout_rate':0.3},
    'B4':{'width_coefficient':1.4,'depth_coefficient':1.8,'dropout_rate':0.4},
    'B5':{'width_coefficient':1.6,'depth_coefficient':2.2,'dropout_rate':0.4},
    'B6':{'width_coefficient':1.8,'depth_coefficient':2.6,'dropout_rate':0.5},
    'B7':{'width_coefficient':2.0,'depth_coefficient':3.1,'dropout_rate':0.5},
    }


CONV_KERNEL_INITIALIZER=["VarianceScaling",2.0,"fan_out","truncated_normal"]
DENSE_KERNEL_INITIALIZER=["VarianceScaling",1.0/3.0,"fan_out","uniform"]
IMAGENET_STDDEV_RGB = [0.229, 0.224, 0.225]