import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class DenseLayer:
    def __init__(self, input_size, growth_rate, dtype='float32'):
        self.growth_rate = growth_rate
        self.weight1 = initializer([1, 1, input_size, 4*self.growth_rate], 'Xavier', dtype)
        self.weight2 = initializer([3, 3, 4*self.growth_rate, self.growth_rate], 'Xavier', dtype)
        self.param = [self.weight1, self.weight2]
    
    
    def output(self, inputs, train_flag=True):
        mean=tf.math.reduce_mean(inputs, axis=3)
        variance=tf.math.reduce_variance(inputs, axis=3)
        mean=tf.expand_dims(mean,axis=-1)
        variance=tf.expand_dims(variance,axis=-1)
        x = tf.nn.batch_normalization(inputs,
                                      mean=mean,
                                      variance=variance,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=1e-3)
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weight1, strides=1, padding="SAME")
        mean=tf.math.reduce_mean(x, axis=3)
        variance=tf.math.reduce_variance(x, axis=3)
        mean=tf.expand_dims(mean,axis=-1)
        variance=tf.expand_dims(variance,axis=-1)
        x = tf.nn.batch_normalization(x,
                              mean=mean,
                              variance=variance,
                              offset=None,
                              scale=None,
                              variance_epsilon=1e-3)
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weight2, strides=1, padding="SAME")
        x = tf.concat([inputs, x], axis=3)
        return x


class DenseBlock:
    def __init__(self, input_size, num_layers, growth_rate, dtype='float32'):
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = []
        self.param = []
        for i in range(self.num_layers):
            self.layers.append(DenseLayer(input_size, growth_rate, dtype))
            self.param.append(self.layers[i].param)
        self.output_size = input_size+growth_rate
    
    
    def output(self, inputs, train_flag=True):
        x = inputs
        for layer in self.layers:
            output = layer.output(x, train_flag=train_flag)
        return output


class TransitionLayer:
    def __init__(self, input_size, compression_factor, dtype='float32'):
        self.compression_factor = compression_factor
        self.weight = initializer([1, 1, input_size, int(self.compression_factor * input_size)], 'Xavier', dtype)
        self.param = [self.weight]
        self.output_size = int(self.compression_factor * input_size)
    
    
    def output(self, inputs, train_flag=True):
        mean=tf.math.reduce_mean(inputs, axis=3)
        variance=tf.math.reduce_variance(inputs, axis=3)
        mean=tf.expand_dims(mean,axis=-1)
        variance=tf.expand_dims(variance,axis=-1)
        x = tf.nn.batch_normalization(inputs,
                                      mean=mean,
                                      variance=variance,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=1e-3)
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weight, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.avg_pool2d(x, ksize=[2, 2], strides=[2, 2], padding="SAME")
        return x


class DenseNet169:
    def __init__(self, input_size, num_classes, growth_rate=32, compression_factor=0.5, include_top=True, pooling=None, dtype='float32'):
        self.conv1_weight = initializer([7, 7, 3, 64], 'Xavier', dtype)
        self.input_size=input_size
        self.num_classes=num_classes
        self.growth_rate=growth_rate
        self.compression_factor=compression_factor
        self.include_top=include_top
        self.pooling=pooling
        self.loss_object=tf.keras.losses.CategoricalCrossentropy() # create a categorical crossentropy loss object
        self.optimizer=Adam() # create an Adam optimizer object
        self.dtype=dtype
        self.km=0
    
    
    def build(self):
        self.bc=tf.Variable(0,dtype=self.dtype)
        self.block1 = DenseBlock(input_size=self.input_size,num_layers=6,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        
        self.trans1 = TransitionLayer(input_size=self.block1.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
        
        self.block2 = DenseBlock(input_size=self.trans1.output_size,num_layers=12,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        
        self.trans2 = TransitionLayer(input_size=self.block2.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
        
        self.block3 = DenseBlock(input_size=self.trans2.output_size,num_layers=32,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        
        self.trans3 = TransitionLayer(input_size=self.block3.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
        
        self.block4 = DenseBlock(input_size=self.trans3.output_size,num_layers=32,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        self.fc_weight = initializer([self.block4.output_size, self.num_classes], 'Xavier', self.dtype)
        self.param=[self.block1.param,self.trans1.param,self.block2.param,self.trans2.param,
                    self.block3.param,self.trans3.param,self.block4.param,self.conv1_weight,
                    ]
        return
    
    
    def fp(self, data, p):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                x = tf.pad(data, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
                x = tf.nn.conv2d(x, self.conv1_weight, strides=2, padding="VALID")
                mean = tf.math.reduce_mean(x, axis=3)
                variance = tf.math.reduce_variance(x, axis=3)
                mean=tf.expand_dims(mean,axis=-1)
                variance=tf.expand_dims(variance,axis=-1)
                x = tf.nn.batch_normalization(x,
                                              mean=mean,
                                              variance=variance,
                                              offset=None,
                                              scale=None,
                                              variance_epsilon=1e-5)
                x = tf.nn.relu(x)
                x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
                x = tf.nn.max_pool2d(x, ksize=[3, 3], strides=[2, 2], padding="VALID")
                x = self.block1.output(x)
                x = self.trans1.output(x)
                x = self.block2.output(x)
                x = self.trans2.output(x)
                x = self.block3.output(x)
                x = self.trans3.output(x)
                x = self.block4.output(x)
                mean=tf.math.reduce_mean(x, axis=3)
                variance=tf.math.reduce_variance(x, axis=3)
                mean=tf.expand_dims(mean,axis=-1)
                variance=tf.expand_dims(variance,axis=-1)
                x = tf.nn.batch_normalization(x,
                                              mean=tf.math.reduce_mean(x),
                                              variance=tf.math.reduce_variance(x),
                                              offset=None,
                                              scale=None,
                                              variance_epsilon=1e-3)
                x = tf.nn.relu(x)
                if self.include_top:
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                    x = tf.matmul(x, self.fc_weight)
                    x = tf.nn.softmax(x)
                else:
                    if self.pooling=="avg":
                        x = tf.math.reduce_mean(x, axis=[1, 2])
                    elif self.pooling=="max":
                        x = tf.math.reduce_max(x, axis=[1, 2])
        else:
            x = tf.nn.conv2d(data, self.conv1_weight, strides=[1, 2, 2, 1], padding="SAME")
            x = tf.nn.max_pool2d(x, ksize=[3, 3], strides=[2, 2], padding="SAME")
            x = self.block1.output(x)
            x = self.trans1.output(x)
            x = self.block2.output(x)
            x = self.trans2.output(x)
            x = self.block3.output(x)
            x = self.trans3.output(x)
            x = self.block4.output(x)
            x = tf.nn.relu(x)
            if self.include_top:
                x = tf.math.reduce_mean(x, axis=[1, 2])
                x = tf.matmul(x, self.fc_weight)
                x = tf.nn.softmax(x)
            else:
                if self.pooling=="avg":
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                elif self.pooling=="max":
                    x = tf.math.reduce_max(x, axis=[1, 2])
        return x
    
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,'GPU')): # assign the device to use
            loss_value=self.loss_object(labels,output) # calculate the loss value using categorical crossentropy loss function
        return loss_value # return the loss value
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,'GPU')): # assign the device to use
            param=self.optimizer.opt(gradient,self.param,self.bc[0]) # update the model parameters using Adam optimizer and batch count
            return param # return the updated model parameters
