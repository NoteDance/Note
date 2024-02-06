import tensorflow as tf
import numpy as np
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization
from Note.nn.Layers import Layers
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Module import Module


class ConvNeXtBlock:
    def __init__(self, input_channels, projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, dtype='float32'):
        self.conv2d=conv2d(projection_dim,[7,7],input_channels//projection_dim,padding='SAME',dtype=dtype)
        self.layer_normalization=layer_normalization(self.conv2d.output_size,dtype=dtype)
        self.dense1=dense(4*projection_dim,projection_dim,activation='gelu',dtype=dtype)
        self.dense2=dense(projection_dim,4*projection_dim,dtype=dtype)
        self.gamma=tf.Variable(tf.ones([projection_dim],dtype=dtype)*layer_scale_init_value)
        self.drop_path_rate=drop_path_rate
        self.layer_scale_init_value=layer_scale_init_value
        self.output_size=projection_dim
        
    
    def LayerScale(self,x):
        return x * self.gamma
    
    
    def StochasticDepth(self,x,train_flag=True):
        if train_flag:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    
    def output(self,data,train_flag=True):
        x=self.conv2d.output(data)
        x=self.layer_normalization.output(x)
        x=self.dense1.output(x)
        x=self.dense2.output(x)
        if self.layer_scale_init_value is not None:
            x=self.LayerScale(x)
        if self.drop_path_rate:
            x=self.StochasticDepth(x,train_flag)
        else:
            x=x
        return data + x


class ConvNeXt:
    def __init__(self,model_type='base',drop_path_rate=0.0,layer_scale_init_value=1e-6,classes=1000,classifier_activation="softmax",include_top=True,pooling=None,device='GPU'):
        self.model_type=model_type
        self.classes=classes
        self.classifier_activation=classifier_activation
        self.depths=MODEL_CONFIGS[model_type]['depths']
        self.projection_dims=MODEL_CONFIGS[model_type]['projection_dims']
        self.drop_path_rate=drop_path_rate
        self.layer_scale_init_value=layer_scale_init_value
        self.include_top=include_top
        self.pooling=pooling
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.km=0
    
    
    def build(self,dtype='float32'):
        Module.init()
        
        # Stem block.
        layers=Layers()
        layers.add(conv2d(self.projection_dims[0],[4,4],3,dtype=dtype))
        layers.add(layer_normalization(dtype=dtype))
        
        # Downsampling blocks.
        self.downsample_layers = []
        self.downsample_layers.append(layers)
        
        num_downsample_layers = 3
        for i in range(num_downsample_layers):
            layers=Layers()
            layers.add(layer_normalization(self.projection_dims[i],dtype=dtype))
            layers.add(conv2d(self.projection_dims[i+1],[2,2],self.projection_dims[i],dtype=dtype))
            self.downsample_layers.append(layers)
        
        # Stochastic depth schedule.
        # This is referred from the original ConvNeXt codebase:
        # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
        depth_drop_rates = [
            float(x) for x in np.linspace(0.0, self.drop_path_rate, sum(self.depths))
        ]
        
        # First apply downsampling blocks and then apply ConvNeXt stages.
        cur = 0
        
        self.blocks=[]
        self.num_convnext_blocks = 4
        for i in range(self.num_convnext_blocks):
            input_channels=self.downsample_layers[i].output_size
            for j in range(self.depths[i]):
                block = Layers()
                block.add(ConvNeXtBlock(
                    input_channels=input_channels,
                    projection_dim=self.projection_dims[i],
                    drop_path_rate=depth_drop_rates[cur + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    dtype=dtype
                    ))
                input_channels=block.output_size
            self.blocks.append(block)
            cur += self.depths[i]
        self.layer_normalization=layer_normalization(self.blocks[-1].output_size,dtype=dtype)
        self.dense=dense(self.classes,self.blocks[-1].output_size,activation=self.classifier_activation,dtype=dtype)
        
        self.dtype=dtype
        self.optimizer=Adam()
        self.param=Module.param
        return
    
    
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
            self.dense_=self.dense
            self.dense=dense(classes,self.dense.input_size,activation=self.classifier_activation,dtype=self.dense.dtype)
            param.extend(self.dense.param)
            self.param=param
            self.optimizer_=self.optimizer
            self.optimizer=Adam(lr=lr,param=self.param)
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
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                for i in range(self.num_convnext_blocks):
                    data = self.downsample_layers[i].output(data)
                    for j in range(self.depths[i]):
                        data=self.blocks[i].output(data)
                if self.include_top:
                    data = tf.math.reduce_mean(data, axis=[1, 2])
                    data = self.layer_normalization.output(data)
                    data = self.dense.output(data)
                else:
                    if self.pooling=="avg":
                        data = tf.math.reduce_mean(data, axis=[1, 2])
                    elif self.pooling=="max":
                        data = tf.math.reduce_max(data, axis=[1, 2])
        else:
            for i in range(self.num_convnext_blocks):
                data = self.downsample_layers[i].output(data,self.km)
                for j in range(self.depths[i]):
                    data=self.blocks[i].output(data,self.km)
            if self.include_top:
                data = tf.math.reduce_mean(data, axis=[1, 2])
                data = self.layer_normalization.output(data)
                data = self.dense.output(data)
            else:
                if self.pooling=="avg":
                    data = tf.math.reduce_mean(data, axis=[1, 2])
                elif self.pooling=="max":
                    data = tf.math.reduce_max(data, axis=[1, 2])
        return data
    
    
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
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param


MODEL_CONFIGS = {
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
    },
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
    },
}