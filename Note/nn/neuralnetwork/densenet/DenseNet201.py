import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


# define a class for dense layer
class DenseLayer:
    # initialize the class with input size, growth rate and data type
    def __init__(self, input_size, growth_rate, dtype='float32'):
        self.growth_rate = growth_rate
        # initialize the first convolutional weight with Xavier initialization
        self.weight1 = initializer([1, 1, input_size, 4*self.growth_rate], 'Xavier', dtype)
        # initialize the second convolutional weight with Xavier initialization
        self.weight2 = initializer([3, 3, 4*self.growth_rate, self.growth_rate], 'Xavier', dtype)
        # store the parameters in a list
        self.param = [self.weight1, self.weight2]
    
    
    # define a method for outputting the layer result
    def output(self, inputs, train_flag=True):
        # calculate the mean and variance of the inputs along the channel axis
        mean=tf.math.reduce_mean(inputs, axis=3)
        variance=tf.math.reduce_variance(inputs, axis=3)
        # expand the mean and variance dimensions to match the inputs shape
        mean=tf.expand_dims(mean,axis=-1)
        variance=tf.expand_dims(variance,axis=-1)
        # perform batch normalization on the inputs with no offset and scale
        x = tf.nn.batch_normalization(inputs,
                                      mean=mean,
                                      variance=variance,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=1e-3)
        # apply relu activation function on the normalized inputs
        x = tf.nn.relu(x)
        # perform the first convolution operation with same padding and stride 1
        x = tf.nn.conv2d(x, self.weight1, strides=1, padding="SAME")
        # calculate the mean and variance of the first convolution result along the channel axis
        mean=tf.math.reduce_mean(x, axis=3)
        variance=tf.math.reduce_variance(x, axis=3)
        # expand the mean and variance dimensions to match the first convolution result shape
        mean=tf.expand_dims(mean,axis=-1)
        variance=tf.expand_dims(variance,axis=-1)
        # perform batch normalization on the first convolution result with no offset and scale
        x = tf.nn.batch_normalization(x,
                              mean=mean,
                              variance=variance,
                              offset=None,
                              scale=None,
                              variance_epsilon=1e-3)
        # apply relu activation function on the normalized first convolution result
        x = tf.nn.relu(x)
        # perform the second convolution operation with same padding and stride 1
        x = tf.nn.conv2d(x, self.weight2, strides=1, padding="SAME")
        # concatenate the inputs and the second convolution result along the channel axis
        x = tf.concat([inputs, x], axis=3)
        # return the output of the dense layer
        return x


# define a class for dense block
class DenseBlock:
    # initialize the class with input size, number of layers, growth rate and data type
    def __init__(self, input_size, num_layers, growth_rate, dtype='float32'):
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = []
        self.param = []
        # create a list of dense layers with different input sizes according to the growth rate
        for i in range(self.num_layers):
            self.layers.append(DenseLayer(input_size, growth_rate, dtype))
            self.param.append(self.layers[i].param)
            input_size += growth_rate
        # store the output size of the dense block as an attribute
        self.output_size = input_size
    
    
    # define a method for outputting the block result
    def output(self, inputs, train_flag=True):
        x = inputs
        # loop through each layer in the block and get its output as the input for the next layer
        for layer in self.layers:
            x = layer.output(x, train_flag=train_flag)
        # return the output of the dense block
        return x


# define a class for transition layer
class TransitionLayer:
    # initialize the class with input size, compression factor and data type
    def __init__(self, input_size, compression_factor, dtype='float32'):
        self.compression_factor = compression_factor
        # initialize the convolutional weight with Xavier initialization
        self.weight = initializer([1, 1, input_size, int(self.compression_factor * input_size)], 'Xavier', dtype)
        # store the parameter in a list
        self.param = [self.weight]
        # store the output size of the transition layer as an attribute
        self.output_size = int(self.compression_factor * input_size)
    
    
    # define a method for outputting the layer result
    def output(self, inputs, train_flag=True):
        # calculate the mean and variance of the inputs along the channel axis
        mean=tf.math.reduce_mean(inputs, axis=3)
        variance=tf.math.reduce_variance(inputs, axis=3)
        # expand the mean and variance dimensions to match the inputs shape
        mean=tf.expand_dims(mean,axis=-1)
        variance=tf.expand_dims(variance,axis=-1)
        # perform batch normalization on the inputs with no offset and scale
        x = tf.nn.batch_normalization(inputs,
                                      mean=mean,
                                      variance=variance,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=1e-3)
        # apply relu activation function on the normalized inputs
        x = tf.nn.relu(x)
        # perform the convolution operation with same padding and stride 1
        x = tf.nn.conv2d(x, self.weight, strides=[1, 1, 1, 1], padding="SAME")
        # perform average pooling operation with kernel size 2 and stride 2
        x = tf.nn.avg_pool2d(x, ksize=[2, 2], strides=[2, 2], padding="SAME")
        # return the output of the transition layer
        return x


# define a class for DenseNet201 model
class DenseNet201:
    # initialize the class with input size, number of classes, growth rate, compression factor,
    # include top flag, pooling option and data type
    def __init__(self, input_size, num_classes=1000, growth_rate=32, compression_factor=0.5, include_top=True, pooling=None, dtype='float32'):
        # initialize the first convolutional weight with Xavier initialization
        self.conv1_weight = initializer([7, 7, 3, 64], 'Xavier', dtype)
        self.input_size=input_size
        self.num_classes=num_classes
        self.growth_rate=growth_rate
        self.compression_factor=compression_factor
        self.include_top=include_top
        self.pooling=pooling
        # initialize the loss object as categorical crossentropy
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        # initialize the optimizer as Adam optimizer
        self.optimizer=Adam()
        self.dtype=dtype
        self.km=0
    
    
    # define a method for building the model parameters
    def build(self):
        # initialize a variable for batch count as zero
        self.bc=tf.Variable(0,dtype=self.dtype)
        
        # create a dense block with input size equal to input size attribute,
        # number of layers equal to 6 and growth rate equal to growth rate attribute
        self.block1 = DenseBlock(input_size=self.input_size,num_layers=6,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        
        # create a transition layer with input size equal to output size of block1,
        # and compression factor equal to compression factor attribute
        self.trans1 = TransitionLayer(input_size=self.block1.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
        
         # create a dense block with input size equal to output size of trans1,
         # number of layers equal to 12 and growth rate equal to growth rate attribute       
        self.block2 = DenseBlock(input_size=self.trans1.output_size,num_layers=12,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        
         # create a transition layer with input size equal to output size of block2,
         # and compression factor equal to compression factor attribute       
        self.trans2 = TransitionLayer(input_size=self.block2.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
        
         # create a dense block with input size equal to output size of trans2,
         # number of layers equal to 48 and growth rate equal to growth rate attribute       
        self.block3 = DenseBlock(input_size=self.trans2.output_size,num_layers=48,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        
        # create a transition layer with input size equal to output size of block3,
        # and compression factor equal to compression factor attribute
        self.trans3 = TransitionLayer(input_size=self.block3.output_size,compression_factor=self.compression_factor,
                                      dtype=self.dtype)
        
         # create a dense block with input size equal to output size of trans3,
         # number of layers equal to 32 and growth rate equal to growth rate attribute
        self.block4 = DenseBlock(input_size=self.trans3.output_size,num_layers=32,
                                 growth_rate=self.growth_rate,
                                 dtype=self.dtype)
        # initialize the fully connected weight with Xavier initialization
        self.fc_weight = initializer([self.block4.output_size, self.num_classes], 'Xavier', self.dtype)
        # store the parameters of all the blocks and layers in a list
        self.param=[self.block1.param,self.trans1.param,self.block2.param,self.trans2.param,
                    self.block3.param,self.trans3.param,self.block4.param,self.conv1_weight,
                    ]
        return
    
    
    # define a method for forward propagation
    def fp(self, data, p):
        # check if kernel mode is 1
        if self.km==1:
            # assign the device for parallel computation
            with tf.device(assign_device(p,'GPU')):
                # pad the data with 3 pixels on each side
                x = tf.pad(data, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
                # perform the first convolution operation with valid padding and stride 2
                x = tf.nn.conv2d(x, self.conv1_weight, strides=2, padding="VALID")
                # calculate the mean and variance of the first convolution result along the channel axis
                mean = tf.math.reduce_mean(x, axis=3)
                variance = tf.math.reduce_variance(x, axis=3)
                # expand the mean and variance dimensions to match the first convolution result shape
                mean=tf.expand_dims(mean,axis=-1)
                variance=tf.expand_dims(variance,axis=-1)
                # perform batch normalization on the first convolution result with no offset and scale
                x = tf.nn.batch_normalization(x,
                                              mean=mean,
                                              variance=variance,
                                              offset=None,
                                              scale=None,
                                              variance_epsilon=1e-5)
                # apply relu activation function on the normalized first convolution result
                x = tf.nn.relu(x)
                # pad the result with 1 pixel on each side
                x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
                # perform max pooling operation with kernel size 3 and stride 2
                x = tf.nn.max_pool2d(x, ksize=[3, 3], strides=[2, 2], padding="VALID")
                # get the output of the first dense block
                x = self.block1.output(x)
                # get the output of the first transition layer
                x = self.trans1.output(x)
                # get the output of the second dense block
                x = self.block2.output(x)
                # get the output of the second transition layer
                x = self.trans2.output(x)
                # get the output of the third dense block
                x = self.block3.output(x)
                # get the output of the third transition layer
                x = self.trans3.output(x)
                # get the output of the fourth dense block
                x = self.block4.output(x)
                # calculate the mean and variance of the fourth dense block result along the channel axis
                mean=tf.math.reduce_mean(x, axis=3)
                variance=tf.math.reduce_variance(x, axis=3)
                # expand the mean and variance dimensions to match the fourth dense block result shape
                mean=tf.expand_dims(mean,axis=-1)
                variance=tf.expand_dims(variance,axis=-1)
                # perform batch normalization on the fourth dense block result with no offset and scale
                x = tf.nn.batch_normalization(x,
                                              mean=tf.math.reduce_mean(x),
                                              variance=tf.math.reduce_variance(x),
                                              offset=None,
                                              scale=None,
                                              variance_epsilon=1e-3)
                # apply relu activation function on the normalized fourth dense block result
                x = tf.nn.relu(x)
                # check if the model includes a fully connected layer
                if self.include_top:
                    # perform global average pooling on the result
                    x = tf.reduce_mean(x, axis=[1, 2])
                    # perform matrix multiplication with the fully connected weight
                    x = tf.matmul(x, self.fc_weight)
                    # apply softmax activation function on the result to get the class probabilities
                    x = tf.nn.softmax(x)
                    # return the output of the model
                    return x
                else:
                    # check the pooling option of the model
                    if self.pooling == 'avg':
                        # perform global average pooling on the result
                        x = tf.reduce_mean(x, axis=[1, 2])
                        # return the output of the model
                        return x
                    elif self.pooling == 'max':
                        # perform global max pooling on the result
                        x = tf.reduce_max(x, axis=[1, 2])
                        # return the output of the model
                        return x
                    else:
                        # return the output of the model without pooling
                        return x
        else:
            # perform the first convolution operation with same padding and stride 2
            x = tf.nn.conv2d(data, self.conv1_weight, strides=[1, 2, 2, 1], padding="SAME")
            # perform max pooling operation with kernel size 3 and stride 2
            x = tf.nn.max_pool2d(x, ksize=[3, 3], strides=[2, 2], padding="SAME")
            # get the output of the first dense block
            x = self.block1.output(x)
            # get the output of the first transition layer
            x = self.trans1.output(x)
            # get the output of the second dense block
            x = self.block2.output(x)
            # get the output of the second transition layer
            x = self.trans2.output(x)
            # get the output of the third dense block
            x = self.block3.output(x)
            # get the output of the third transition layer
            x = self.trans3.output(x)
            # get the output of the fourth dense block
            x = self.block4.output(x)
            # apply relu activation function on the result
            x = tf.nn.relu(x)
            # check if the model includes a fully connected layer
            if self.include_top:
                # perform global average pooling on the result
                x = tf.math.reduce_mean(x, axis=[1, 2])
                # perform matrix multiplication with the fully connected weight
                x = tf.matmul(x, self.fc_weight)
                # apply softmax activation function on the result to get the class probabilities
                x = tf.nn.softmax(x)
            else:
                # check the pooling option of the model
                if self.pooling=="avg":
                    # perform global average pooling on the result
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                elif self.pooling=="max":
                    # perform global max pooling on the result
                    x = tf.math.reduce_max(x, axis=[1, 2])
        # return the output of the model        
        return x
    
    
    # define a method for calculating the loss value
    def loss(self,output,labels,p):
        # assign the device for parallel computation
        with tf.device(assign_device(p,'GPU')):
            # calculate the categorical crossentropy loss between output and labels 
            loss_value=self.loss_object(labels,output)
        # return the loss value    
        return loss_value
    
    
    # define a method for applying the optimizer
    def opt(self,gradient,p):
        # assign the device for parallel computation
        with tf.device(assign_device(p,'GPU')):
            # update the parameters with the gradient and the batch count using the Adam optimizer
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            # return the updated parameters
            return param
