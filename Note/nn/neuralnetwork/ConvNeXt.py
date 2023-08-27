import tensorflow as tf
import numpy as np
from Note.nn.initializer import initializer
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization
from Note.nn.Layers import Layers
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class ConvNeXtBlock:
    def __init__(self, input_channels, projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, dtype='float32'):
        self.conv2d=conv2d([7,7,input_channels//projection_dim,projection_dim],padding='SAME',dtype=dtype)
        self.layer_normalization=layer_normalization()
        self.dense1=dense([projection_dim,4*projection_dim],activation='gelu',dtype=dtype)
        self.dense2=dense([4*projection_dim,projection_dim],dtype=dtype)
        self.gamma=tf.Variable(tf.ones([projection_dim],dtype=dtype)*layer_scale_init_value)
        self.drop_path_rate=drop_path_rate
        self.layer_scale_init_value=layer_scale_init_value
        self.output_size=projection_dim
        self.param=[self.conv2d.param,self.dense1.param,self.dense2.param,self.gamma]
        
    
    def LayerScale(self,x):
        return x * self.gamma
    
    
    def StochasticDepth(self,x,train_flag=True):
        if train_flag:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    
    def output(self,data,train_flag=True):
        x=self.conv2d.output(data)
        x=self.layer_normalization.output(x,train_flag=train_flag)
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
    def __init__(self,input_channels,model_type='base',drop_path_rate=0.0,layer_scale_init_value=1e-6,classes=1000,include_top=True,pooling=None):
        self.input_channels=input_channels
        self.model_type=model_type
        self.classes=classes
        self.depths=MODEL_CONFIGS[model_type]['depths']
        self.projection_dims=MODEL_CONFIGS[model_type]['projection_dims']
        self.drop_path_rate=drop_path_rate
        self.layer_scale_init_value=layer_scale_init_value
        self.include_top=include_top
        self.pooling=pooling
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.optimizer=Adam()
        self.param=[]
        self.km=0
    
    
    def build(self,dtype='float32'):
        self.bc=tf.Variable(0,dtype=dtype)
        # Stem block.
        layers=Layers()
        layers.add(conv2d([4,4,self.input_channels,self.projection_dims[0]],dtype=dtype))
        layers.add(layer_normalization())
        self.param.append(layers.param)
        
        # Downsampling blocks.
        self.downsample_layers = []
        self.downsample_layers.append(layers)
        
        num_downsample_layers = 3
        for i in range(num_downsample_layers):
            layers=Layers()
            layers.add(layer_normalization())
            layers.add(conv2d([2,2,self.projection_dims[i],self.projection_dims[i+1]],dtype=dtype))
            self.downsample_layers.append(layers)
            self.param.append(layers.param)
        
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
            self.blocks.append([])
            input_channels=self.downsample_layers[i].output_size
            for j in range(self.depths[i]):
                block = ConvNeXtBlock(
                    input_channels=input_channels,
                    projection_dim=self.projection_dims[i],
                    drop_path_rate=depth_drop_rates[cur + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    dtype=dtype
                    )
                self.blocks[i].append(block)
                input_channels=block.output_size
                self.param.append(block.param)
            cur += self.depths[i]
        self.layer_normalization=layer_normalization()
        self.fc_weight = initializer([self.blocks[-1][-1].output_size, self.classes], 'Xavier', dtype)
        self.fc_bias = initializer([self.classes], 'Xavier', dtype)
        self.param.extend([self.fc_weight,self.fc_bias])
        return
    
    
    def fp(self,data,p):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                for i in range(self.num_convnext_blocks):
                    data = self.downsample_layers[i].output(data)
                    for j in range(self.depths[i]):
                        data=self.blocks[i][j].output(data)
                if self.include_top:
                    data = tf.math.reduce_mean(data, axis=[1, 2])
                    data = self.layer_normalization.output(data)
                    data = tf.matmul(data, self.fc_weight) + self.fc_bias
                    data = tf.nn.softmax(data)
                else:
                    if self.pooling=="avg":
                        data = tf.math.reduce_mean(data, axis=[1, 2])
                    elif self.pooling=="max":
                        data = tf.math.reduce_max(data, axis=[1, 2])
        else:
            for i in range(self.num_convnext_blocks):
                data = self.downsample_layers[i].output(data,self.km)
                for j in range(self.depths[i]):
                    data=self.blocks[i][j].output(data,self.km)
            if self.include_top:
                data = tf.math.reduce_mean(data, axis=[1, 2])
                data = tf.matmul(data, self.fc_weight) + self.fc_bias
                data = tf.nn.softmax(data)
            else:
                if self.pooling=="avg":
                    data = tf.math.reduce_mean(data, axis=[1, 2])
                elif self.pooling=="max":
                    data = tf.math.reduce_max(data, axis=[1, 2])
        return data
    
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,'GPU')):
            loss_value=self.loss_object(labels,output)
        return loss_value
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,'GPU')):
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
