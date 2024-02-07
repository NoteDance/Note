import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_norm import batch_norm
from Note.nn.layer.dropout import dropout
from Note.nn.layer.global_avg_pool2d import global_avg_pool2d
from Note.nn.layer.global_max_pool2d import global_max_pool2d
from Note.nn.layer.norm import norm
from Note.nn.layer.image_preprocessing.rescaling import rescaling
from Note.nn.layer.multiply import multiply
from Note.nn.layer.reshape import reshape
from Note.nn.layer.identity import identity
from Note.nn.Layers import Layers
from Note.nn.activation import activation_dict
import copy
from Note.nn.Module import Module


class EfficientNetV2:
    """Instantiates the EfficientNetV2 architecture using given scaling
    coefficients.

    Args:
      input_shape: optional shape tuple, only to be specified if `include_top`
        is False. It should have exactly 3 inputs channels.
      model_name: string, model name.
      dropout_rate: float, dropout rate before final classifier layer.
      drop_connect_rate: float, dropout rate at skip connections.
      depth_divisor: integer, a unit of network width.
      min_depth: integer, minimum number of filters.
      bn_momentum: float. Momentum parameter for Batch Normalization layers.
      activation: activation function.
      blocks_args: list of dicts, parameters to construct block modules.
      include_top: whether to include the fully-connected layer at the top of
        the network.
      weights: one of `None` (random initialization), `"imagenet"` (pre-training
        on ImageNet), or the path to the weights file to be loaded.
      pooling: optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor output
          of the last convolutional layer.
        - "avg" means that global average pooling will be applied to the output
          of the last convolutional layer, and thus the output of the model will
          be a 2D tensor.
        - `"max"` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified.
      classifier_activation: A string or callable. The activation function to
        use on the `"top"` layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the `"top"` layer.
      include_preprocessing: Boolean, whether to include the preprocessing layer
        (`Rescaling`) at the bottom of the network. Defaults to `True`.
        
    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `"softmax"` or `None` when
        using a pretrained top layer.
    """
    def __init__(
                self,
                input_shape,
                model_name="efficientnetv2-b0",
                dropout_rate=0.2,
                drop_connect_rate=0.2,
                depth_divisor=8,
                min_depth=8,
                bn_momentum=0.9,
                activation="swish",
                blocks_args="default",
                include_top=True,
                weights="imagenet",
                pooling=None,
                classes=1000,
                classifier_activation="softmax",
                include_preprocessing=True,
                dtype='float32'
            ):
        if include_preprocessing:
            # Apply original V1 preprocessing for Bx variants
            if model_name.split("-")[-1].startswith("b"):
                self.rescaling=rescaling(scale=1.0 / 255)
                self.norm=norm(
                    input_shape,
                    mean=[0.485, 0.456, 0.406],
                    variance=[0.229**2, 0.224**2, 0.225**2],
                    axis=-1,
                )
            else:
                self.rescaling=rescaling(scale=1.0 / 128.0, offset=-1)
                
        if blocks_args == "default":
            self.blocks_args = DEFAULT_BLOCKS_ARGS[model_name]
    
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
        self.dropout_rate=dropout_rate
        self.drop_connect_rate=drop_connect_rate
        self.depth_divisor=depth_divisor
        self.min_depth=min_depth
        self.bn_momentum=bn_momentum
        self.activation=activation
        self.classes=classes # store the number of classes
        self.include_top=include_top
        self.pooling=pooling
        self.classifier_activation=classifier_activation
        self.dtype=dtype
        self.norm=norm(input_shape,dtype=dtype)
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy() # create a sparse categorical crossentropy loss object
        self.km=0
    
    
    def build(self):
        Module.init()
        
        def round_filters(filters, width_coefficient, min_depth, depth_divisor):
            """Round number of filters based on depth multiplier."""
            filters *= width_coefficient
            minimum_depth = min_depth or depth_divisor
            new_filters = max(
                minimum_depth,
                int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
            )
            return int(new_filters)
        
        
        def round_repeats(repeats, depth_coefficient):
            """Round number of repeats based on depth multiplier."""
            return int(tf.math.ceil(depth_coefficient * repeats))
    
        # Build stem
        stem_filters = round_filters(
            filters=self.blocks_args[0]["input_filters"],
            width_coefficient=self.width_coefficient,
            min_depth=self.min_depth,
            depth_divisor=self.depth_divisor,
        )
        self.layers1=Layers()
        self.layers1.add(conv2d(stem_filters,[3,3],3,strides=2,padding="SAME",use_bias=False,
                           weight_initializer=CONV_KERNEL_INITIALIZER,dtype=self.dtype))
        self.layers1.add(batch_norm(axis=-1,momentum=self.bn_momentum,parallel=False,dtype=self.dtype))
        self.layers1.add(activation_dict[self.activation])    
    
        # Build blocks
        blocks_args = copy.deepcopy(self.blocks_args)
        b = 0
        blocks = float(sum(args["num_repeat"] for args in blocks_args))
        self.layers2=Layers()
        for i, args in enumerate(blocks_args):
            assert args["num_repeat"] > 0
    
            # Update block input and output filters based on depth multiplier.
            args["input_filters"] = round_filters(
                filters=args["input_filters"],
                width_coefficient=self.width_coefficient,
                min_depth=self.min_depth,
                depth_divisor=self.depth_divisor,
            )
            args["output_filters"] = round_filters(
                filters=args["output_filters"],
                width_coefficient=self.width_coefficient,
                min_depth=self.min_depth,
                depth_divisor=self.depth_divisor,
            )

            # Determine which conv type to use:
            block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
            repeats = round_repeats(
                repeats=args.pop("num_repeat"), depth_coefficient=self.depth_coefficient
            )
            if i==0:
                in_channels=self.layers1.output_size
            for j in range(repeats):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    args["strides"] = 1
                    args["input_filters"] = args["output_filters"]
                    
                self.layers2.add(block(
                    in_channels,
                    activation=self.activation,
                    bn_momentum=self.bn_momentum,
                    survival_probability=self.drop_connect_rate * b / blocks,
                    **args,
                    dtype=self.dtype
                ))
                b += 1
                in_channels=self.layers2.output_size
        
        # Build top
        top_filters = round_filters(
            filters=1280,
            width_coefficient=self.width_coefficient,
            min_depth=self.min_depth,
            depth_divisor=self.depth_divisor,
        )
        self.layers3=Layers()
        self.layers3.add(conv2d(
            top_filters,
            [1,1],
            self.layers2.output_size,
            padding="SAME",
            use_bias=False,
            weight_initializer=CONV_KERNEL_INITIALIZER,
            dtype=self.dtype
        ))
        self.layers3.add(batch_norm(axis=-1,momentum=self.bn_momentum,parallel=False,dtype=self.dtype))
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
        self.param=Module.param
        self.opt=tf.keras.optimizers.Adam()
        
        
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
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
        data=self.rescaling.output(data)
        data=self.norm.output(data)
        x=self.layers1.output(data,self.km)
        x=self.layers2.output(x,self.km)
        x=self.layers3.output(x,self.km)
        if self.include_top:
            x=self.global_avg_pool2d.output(x)
            if self.dropout_rate > 0:
                x=self.dropout.output(x)
            x=self.dense.output(x)
        else:
            if self.pooling == "avg":
                x=self.global_avg_pool2d.output(x)
            else:
                x=self.global_max_pool2d.output(x)
        return x


    def loss(self,output,labels):
        """A method that calculates the loss value between the output tensor and the labels tensor.

        Args:
            output: tensor, the output data after applying the model.
            labels: tensor, the true labels of the input data.
            p: integer, the index of the device to use.

        Returns:
            loss: tensor, the loss value.
        """
        loss=self.loss_object(labels,output) # calculate the loss value using categorical crossentropy loss function
        return loss # return the loss value


class MBConvBlock:
    def __init__(
            self,
            in_channels,
            input_filters: int,
            output_filters: int,
            expand_ratio=1,
            kernel_size=3,
            strides=1,
            se_ratio=0.0,
            bn_momentum=0.9,
            activation="swish",
            survival_probability: float = 0.8,
            dtype='float32'
            ):
        self.strides = strides
        self.input_filters = input_filters
        self.output_filters = output_filters
        # Expansion phase
        filters = input_filters * expand_ratio
        self.layers=Layers()
        if expand_ratio != 1:
            self.layers.add(conv2d(
                filters=filters,
                kernel_size=[1,1],
                input_size=in_channels,
                strides=1,
                weight_initializer=CONV_KERNEL_INITIALIZER,
                padding="SAME",
                use_bias=False,
                dtype=dtype
            ))
            self.layers.add(batch_norm(
                axis=-1,
                momentum=bn_momentum,
                parallel=False,
                dtype=dtype
            ))
            self.layers.add(activation_dict[activation])
        else:
            self.layers.add(identity(in_channels))
    
        # Depthwise conv
        self.layers.add(depthwise_conv2d(
            kernel_size=[kernel_size,kernel_size],
            strides=strides,
            weight_initializer=CONV_KERNEL_INITIALIZER,
            padding="SAME",
            use_bias=False,
            dtype=dtype
        ))
        self.layers.add(batch_norm(
            axis=-1, momentum=bn_momentum, parallel=False, dtype=dtype
        ))
        self.layers.add(activation_dict[activation],save_data=True)
    
        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            self.layers.add(global_avg_pool2d())
            se_shape = (1, 1, filters)
            self.layers.add(reshape(se_shape))
            self.layers.add(conv2d(
                filters_se,
                [1,1],
                padding="SAME",
                activation=activation,
                weight_initializer=CONV_KERNEL_INITIALIZER,
                dtype=dtype
            ))
            self.layers.add(conv2d(
                filters,
                [1,1],
                padding="SAME",
                activation="sigmoid",
                weight_initializer=CONV_KERNEL_INITIALIZER,
                dtype=dtype
            ),save_data=True)
    
            self.layers.add(multiply(),use_data=True)
    
        # Output phase
        self.layers.add(conv2d(
            filters=output_filters,
            kernel_size=[1,1],
            strides=1,
            weight_initializer=CONV_KERNEL_INITIALIZER,
            padding="SAME",
            use_bias=False,
            dtype=dtype
        ))
        self.layers.add(batch_norm(
            axis=-1, momentum=bn_momentum, parallel=False, dtype=dtype
        ))
    
        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                self.layers.add(dropout(
                    survival_probability,
                    noise_shape=(None, 1, 1, 1),
                ))
        self.output_size=self.layers.output_size
        self.train_flag=True
    
    
    def output(self,data,train_flag=True):
        x=self.layers.output(data,train_flag)
        if self.strides == 1 and self.input_filters == self.output_filters:
            x=tf.math.add(x,data)
        return x


class FusedMBConvBlock:
    def __init__(
        self,
        in_channels,
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation="swish",
        survival_probability: float = 0.8,
        dtype='float32'
        ):
        """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a
        conv2d."""
        self.strides = strides
        self.input_filters = input_filters
        self.output_filters = output_filters
        filters = input_filters * expand_ratio
        self.layers=Layers()
        if expand_ratio != 1:
            self.layers.add(conv2d(
                filters,
                kernel_size=[kernel_size,kernel_size],
                input_size=in_channels,
                strides=strides,
                weight_initializer=CONV_KERNEL_INITIALIZER,
                padding="SAME",
                use_bias=False,
                dtype=dtype
            ))
            self.layers.add(batch_norm(
                axis=-1, momentum=bn_momentum, parallel=False, dtype=dtype
            ))
            self.layers.add(activation_dict[activation],save_data=True)
        else:
            self.layers.add(identity(in_channels))

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            self.layers.add(global_avg_pool2d())
            se_shape = (1, 1, filters)
            self.layers.add(reshape(se_shape))

            self.layers.add(conv2d(
                filters_se,
                [1,1],
                padding="SAME",
                activation=activation,
                weight_initializer=CONV_KERNEL_INITIALIZER,
                dtype=dtype
            ))
            self.layers.add(conv2d(
                filters,
                [1,1],
                padding="SAME",
                activation="sigmoid",
                weight_initializer=CONV_KERNEL_INITIALIZER,
                dtype=dtype
            ),save_data=True)
            
            self.layers.add(multiply(),use_data=True)

        # Output phase:
        self.layers.add(conv2d(
            output_filters,
            kernel_size=[1,1] if expand_ratio != 1 else [kernel_size,kernel_size],
            strides=1 if expand_ratio != 1 else strides,
            weight_initializer=CONV_KERNEL_INITIALIZER,
            padding="SAME",
            use_bias=False,
        ))
        self.layers.add(batch_norm(
            axis=-1, momentum=bn_momentum, parallel=False, dtype=dtype
        ))
        if expand_ratio == 1:
            self.layers.add(activation_dict[activation])

        # Residual:
        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                self.layers.add(dropout(
                    survival_probability,
                    noise_shape=(None, 1, 1, 1),
                ))
        self.output_size=self.layers.output_size
        self.train_flag=True
    
    
    def output(self,data,train_flag=True):
        x=self.layers.output(data,train_flag)
        if self.strides == 1 and self.input_filters == self.output_filters:
            x=tf.math.add(x,data)
        return x


DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-s": [
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "conv_type": 1,
            "expand_ratio": 4,
            "input_filters": 48,
            "kernel_size": 3,
            "num_repeat": 4,
            "output_filters": 64,
            "se_ratio": 0,
            "strides": 2,
        },
        {
            "conv_type": 0,
            "expand_ratio": 4,
            "input_filters": 64,
            "kernel_size": 3,
            "num_repeat": 6,
            "output_filters": 128,
            "se_ratio": 0.25,
            "strides": 2,
        },
        {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 128,
            "kernel_size": 3,
            "num_repeat": 9,
            "output_filters": 160,
            "se_ratio": 0.25,
            "strides": 1,
        },
        {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 160,
            "kernel_size": 3,
            "num_repeat": 15,
            "output_filters": 256,
            "se_ratio": 0.25,
            "strides": 2,
        },
    ],
    "efficientnetv2-m": [
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}


MODEL_CONFIG={
    "efficientnetv2-b0":{'width_coefficient':1.0,'depth_coefficient':1.0,'dropout_rate':0.2},
    "efficientnetv2-b1":{'width_coefficient':1.0,'depth_coefficient':1.1,'dropout_rate':0.2},
    "efficientnetv2-b2":{'width_coefficient':1.1,'depth_coefficient':1.2,'dropout_rate':0.3},
    "efficientnetv2-b3":{'width_coefficient':1.2,'depth_coefficient':1.4,'dropout_rate':0.3},
    "efficientnetv2-s":{'width_coefficient':1.0,'depth_coefficient':1.0,'dropout_rate':0.4},
    "efficientnetv2-m":{'width_coefficient':1.0,'depth_coefficient':1.0,'dropout_rate':0.4},
    "efficientnetv2-l":{'width_coefficient':1.0,'depth_coefficient':1.0,'dropout_rate':0.5},
    }


CONV_KERNEL_INITIALIZER = ["VarianceScaling",2.0,"fan_out","truncated_normal"]
DENSE_KERNEL_INITIALIZER = ["VarianceScaling",1.0/3.0,"fan_out","uniform"]