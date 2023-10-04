import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


# Define a class for the MBConv block
class MBConv:
    # Initialize the class with the input and output parameters
    def __init__(self, input_channels, output_size, kernel_size=3, strides=1, expand_ratio=1, repeats=1, se_ratio=0.25, rate=0.2, dtype='float32'):
        # Calculate the expanded size by multiplying the input size by the expand ratio
        self.expanded_size = input_channels * expand_ratio
        # Initialize empty lists for storing the weights of the convolution layers
        self.weight_expand = []
        self.weight_depthwise = []
        self.weight_project = []
        self.weight_se_1 = []
        self.weight_se_2 = []
        self.expand_beta = []
        self.expand_gamma = []
        self.depthwise_beta = []
        self.depthwise_gamma = []
        self.project_beta = []
        self.project_gamma = []
        self.expand_moving_mean = []
        self.expand_moving_var = []
        self.depthwise_moving_mean = []
        self.depthwise_moving_var = []
        self.project_moving_mean = []
        self.project_moving_var = []
        # Calculate the number of channels for the squeeze and excite layer
        se_channels = max(1, int(self.expanded_size * se_ratio))
        # Loop over the number of repeats for the block
        for i in range(repeats):
            # If it is the first repeat, use the input size as the input channels for the expand layer
            if i==0:
                # Initialize the weight for the expand layer using a custom initializer from Note.nn module
                self.weight_expand.append(initializer([1, 1, input_channels, self.expanded_size], 'Xavier', dtype))
                # Initialize the weight for the depthwise layer using a custom initializer from Note.nn module
                self.weight_depthwise.append(initializer([kernel_size, kernel_size, self.expanded_size, 1], 'Xavier', dtype))
                # Initialize the weight for the project layer using a custom initializer from Note.nn module
                self.weight_project.append(initializer([1, 1, self.expanded_size, output_size], 'Xavier', dtype))
            # Otherwise, use the output size as the input channels for the expand layer
            else:
                # Initialize the weight for the expand layer using a custom initializer from Note.nn module
                self.weight_expand.append(initializer([1, 1, output_size, self.expanded_size], 'Xavier', dtype))
                # Initialize the weight for the depthwise layer using a custom initializer from Note.nn module
                self.weight_depthwise.append(initializer([kernel_size, kernel_size, self.expanded_size, 1], 'Xavier', dtype))
                # Initialize the weight for the project layer using a custom initializer from Note.nn module
                self.weight_project.append(initializer([1, 1, self.expanded_size, output_size], 'Xavier', dtype))
            # Initialize the weights for the squeeze and excite layers using a custom initializer from Note.nn module
            self.weight_se_1.append(initializer([1, 1, self.expanded_size, se_channels], 'Xavier', dtype))
            self.weight_se_2.append(initializer([1, 1, se_channels, self.expanded_size], 'Xavier', dtype))
            self.expand_beta.append(tf.Variable(tf.zeros([self.expanded_size], dtype)))
            self.expand_gamma.append(tf.Variable(tf.ones([self.expanded_size], dtype)))
            self.depthwise_beta.append(tf.Variable(tf.zeros([self.expanded_size], dtype)))
            self.depthwise_gamma.append(tf.Variable(tf.ones([self.expanded_size], dtype)))
            self.project_beta.append(tf.Variable(tf.zeros([output_size], dtype)))
            self.project_gamma.append(tf.Variable(tf.ones([output_size], dtype)))
            self.expand_moving_mean.append(tf.zeros([self.expanded_size], dtype))
            self.expand_moving_var.append(tf.ones([self.expanded_size], dtype))
            self.depthwise_moving_mean.append(tf.zeros([self.expanded_size], dtype))
            self.depthwise_moving_var.append(tf.ones([self.expanded_size], dtype))
            self.project_moving_mean.append(tf.zeros([output_size], dtype))
            self.project_moving_var.append(tf.ones([output_size], dtype))
        # Assign the strides to a class attribute
        self.strides = strides
        # Assign the expand ratio to a class attribute
        self.expand_ratio = expand_ratio
        # Assign the repeats to a class attribute
        self.repeats = repeats
        # Assign the se ratio to a class attribute
        self.se_ratio = se_ratio
        # Assign the rate to a class attribute
        self.rate = rate
        # Get the swish activation function from the activation_dict in Note.nn module
        self.swish = activation_dict['swish']
        self.b=tf.Variable(0,dtype=dtype)
        self.train_flag=True
        # Assign the output size to a class attribute
        self.output_size = output_size
        # Store all the weights in a list as a class attribute
        self.param = [self.weight_expand, self.weight_depthwise, self.weight_project, self.weight_se_1, self.weight_se_2,
                      self.expand_beta, self.expand_gamma,
                      self.depthwise_beta, self.depthwise_gamma,
                      self.project_beta, self.project_gamma
                      ]
    
    
    # Define a method for computing the output of the block given an input tensor and a training flag
    def output(self, data, train_flag=True):
        self.train_flag=train_flag
        # Initialize a variable b to 0 for calculating the dropout rate later
        self.b.assign(0)
        # Loop over the number of repeats for the block
        for i in range(self.repeats):
            # If it is the first repeat, use the original strides and input tensor for the depthwise convolution layer
            if i == 0:
                strides_i = self.strides
                inputs_i = data
            # Otherwise, use 1 as the strides and the previous output tensor for the depthwise convolution layer
            else:
                strides_i = 1
                inputs_i = x
            # If the expand ratio is not 1, apply an expansion step using a convolution layer with 1x1 filter and no bias term 
            if self.expand_ratio != 1:
                x = tf.nn.conv2d(inputs_i ,self.weight_expand[i] ,strides=[1 ,1 ,1 ,1] ,padding="SAME")
                # If it is training mode, apply batch normalization to normalize the output tensor along its channel dimension 
                if train_flag:
                    mean, var = tf.nn.moments(x, axes=3, keepdims=True)
                    self.expand_moving_mean[i] = self.expand_moving_mean[i] * 0.9 + mean * (1 - 0.9)
                    self.expand_moving_var[i] = self.expand_moving_var[i] * 0.9 + var * (1 - 0.9)
                    x = tf.nn.batch_normalization(x ,self.expand_moving_mean[i] ,self.expand_moving_var[i] ,self.expand_beta[i] ,self.expand_gamma[i] ,1e-3)
                # Apply swish activation function to the output tensor
                x = self.swish(x)
            # Otherwise, skip the expansion step and use the input tensor as the output tensor
            else:
                x = inputs_i
            # Apply a depthwise convolution step using a convolution layer with kernel size x kernel size filter and no bias term 
            x = tf.nn.depthwise_conv2d(x ,self.weight_depthwise[i] ,strides=[1 ,strides_i ,strides_i ,1] ,padding="SAME")
            # If it is training mode, apply batch normalization to normalize the output tensor along its channel dimension 
            if self.train_flag:
                mean, var = tf.nn.moments(x, axes=3, keepdims=True)
                self.depthwise_moving_mean[i] = self.depthwise_moving_mean[i] * 0.9 + mean * (1 - 0.9)
                self.depthwise_moving_var[i] = self.depthwise_moving_var[i] * 0.9 + var * (1 - 0.9)
                x = tf.nn.batch_normalization(x ,self.depthwise_moving_mean[i] ,self.depthwise_moving_var[i] ,self.depthwise_beta[i] ,self.depthwise_gamma[i] ,1e-3)
            # Apply swish activation function to the output tensor
            x = self.swish(x)
            # If the se ratio is positive and less than or equal to 1, apply a squeeze and excite step to enhance the output tensor
            if 0 < self.se_ratio <= 1:
                # Compute the global average pooling of the output tensor along its spatial dimensions
                se_tensor = tf.reduce_mean(x, axis=[1, 2])
                # Reshape the pooled tensor to have a shape of [batch_size, 1, 1, expanded_size]
                se_tensor = tf.reshape(se_tensor, [-1, 1, 1, self.expanded_size])
                # Apply a convolution layer with 1x1 filter and swish activation function to reduce the number of channels to se_channels
                se_tensor = tf.nn.conv2d(se_tensor, self.weight_se_1[i], strides=[1, 1, 1, 1], padding="SAME")
                se_tensor = self.swish(se_tensor)
                # Apply another convolution layer with 1x1 filter and sigmoid activation function to increase the number of channels back to expanded_size
                se_tensor = tf.nn.conv2d(se_tensor, self.weight_se_2[i], strides=[1, 1, 1, 1], padding="SAME")
                se_tensor = tf.nn.sigmoid(se_tensor)
                # Multiply the output tensor and the squeeze and excite tensor element-wise to form a weighted output tensor
                x = tf.multiply(x, se_tensor)
            # Apply a projection step using a convolution layer with 1x1 filter and no bias term to reduce the number of channels to output_size
            x = tf.nn.conv2d(x, self.weight_project[i], strides=[1, 1, 1, 1], padding="SAME")
            # If it is training mode, apply batch normalization to normalize the output tensor along its channel dimension 
            if self.train_flag:
                mean, var = tf.nn.moments(x, axes=3, keepdims=True)
                self.project_moving_mean[i] = self.project_moving_mean[i] * 0.9 + mean * (1 - 0.9)
                self.project_moving_var[i] = self.project_moving_var[i] * 0.9 + var * (1 - 0.9)
                x = tf.nn.batch_normalization(x, self.project_moving_mean[i], self.project_moving_var[i], self.project_beta[i], self.project_gamma[i] ,1e-3)
            # If the input tensor and the output tensor have the same shape, apply a residual connection by adding them element-wise
            if inputs_i.shape == x.shape:
                # If it is training mode, apply dropout to the output tensor with a variable rate that depends on b and the repeats parameter and a noise shape
                if self.train_flag:
                    rate = self.rate * self.b / self.repeats
                    if rate > 0:
                        x = tf.nn.dropout(x, rate=rate, noise_shape=(None, 1, 1, 1))
                    x = tf.add(x, inputs_i)
            # Increment b by 1 for the next iteration
            self.b.assign_add(1)
        # Return the output tensor
        return x


class EfficientNetB5:
    """A class that implements the EfficientNetB5 model for image classification.

    Args:
        classes: integer, the number of classes to predict. Default is 1000.
    """
    def __init__(self,classes=1000,include_top=True,pooling=None):
        self.classes=classes # store the number of classes
        self.include_top=include_top
        self.pooling=pooling
        self.swish=activation_dict['swish'] # get the swish activation function from the activation dictionary
        self.loss_object=tf.keras.losses.CategoricalCrossentropy() # create a categorical crossentropy loss object
        self.km=0
    
    
    def build(self,dtype='float32'):
        """A method that builds the model by creating different layers."""
        self.conv2d=conv2d(160,[3,3],3,strides=[1,2,2,1],padding="SAME",dtype=dtype) # create a conv2d layer with 160 filters and no bias
        self.MBConv1=MBConv(160,48,3,1,1,1,dtype=dtype) # create a MBConv layer with 48 output channels and 1 repeat
        self.MBConv2=MBConv(48,104,3,2,6,8,dtype=dtype) # create a MBConv layer with 104 output channels and 8 repeats
        self.MBConv3=MBConv(104,176,5,2,6,2,dtype=dtype) # create a MBConv layer with 176 output channels and 2 repeats
        self.MBConv4=MBConv(176,304,3,2,6,8,dtype=dtype) # create a MBConv layer with 304 output channels and 8 repeats
        self.MBConv5=MBConv(304,384,5,1,6,8,dtype=dtype) # create a MBConv layer with 384 output channels and 8 repeats
        self.MBConv6=MBConv(384,576,5,2,6,10,dtype=dtype) # create a MBConv layer with 576 output channels and 10 repeats
        self.MBConv7=MBConv(576,960,3,1,6,1,dtype=dtype) # create a MBConv layer with 960 output channels and 1 repeat
        self.conv1x1=conv2d(1280,[1,1],960,strides=[1,1,1,1],padding="SAME",dtype=dtype) # create a conv2d layer with 1280 filters and no bias
        self.dense=dense(self.classes,1280,dtype=dtype) # create a dense layer with self.classes units
        self.param=[self.conv2d.param,
                    self.MBConv1.param,
                    self.MBConv2.param,
                    self.MBConv3.param,
                    self.MBConv4.param,
                    self.MBConv5.param,
                    self.MBConv6.param,
                    self.MBConv7.param,
                    self.conv1x1.param,
                    self.dense.param
                    ] # store all the parameters in a list
        self.optimizer=Adam(param=self.param) # create an Adam optimizer object
        return
    
    
    def fp(self,data,p=None):
        """A method that performs forward propagation on the input data and returns the output tensor.

        Args:
            data: tensor, the input data.
            p: integer, the index of the device to use.

        Returns:
            output: tensor, the output data after applying the model.
        """
        if self.km==1: # if kernel mode is 1
            with tf.device(assign_device(p,'GPU')): # assign the device to use
                data=self.conv2d.output(data) # apply the conv2d layer with strides 2 and same padding
                data=tf.nn.batch_normalization(data,tf.Variable(tf.zeros([160])),tf.Variable(tf.ones([160])),None,None,1e-5) # apply batch normalization to normalize the output
                data=self.swish(data) # apply swish activation function to increase nonlinearity
                data=self.MBConv1.output(data) # apply the MBConv1 layer
                data=self.MBConv2.output(data) # apply the MBConv2 layer
                data=self.MBConv3.output(data) # apply the MBConv3 layer
                data=self.MBConv4.output(data) # apply the MBConv4 layer
                data=self.MBConv5.output(data) # apply the MBConv5 layer
                data=self.MBConv6.output(data) # apply the MBConv6 layer
                data=self.MBConv7.output(data) # apply the MBConv7 layer
                data=self.conv1x1.output(data) # apply the 1x1 convolution layer with strides 1 and same padding
                if self.include_top:
                    data=tf.reduce_mean(data,[1,2]) # apply global average pooling to get the mean value of each channel
                    data=tf.nn.dropout(data,rate=0.2)
                    output=tf.nn.softmax(self.dense.output(data)) # apply the dense layer and softmax activation function to get the probability distribution of each class
                else:
                    if self.pooling=="avg":
                        data=tf.reduce_mean(data,[1,2])
                    elif self.pooling=="max":
                        data=tf.reduce_max(data,[1,2])
        else:
            data=self.conv2d.output(data) # apply the conv2d layer with strides 2 and same padding
            data=self.swish(data) # apply swish activation function to increase nonlinearity
            data=self.MBConv1.output(data,self.km) # apply the MBConv1 layer
            data=self.MBConv2.output(data,self.km) # apply the MBConv2 layer
            data=self.MBConv3.output(data,self.km) # apply the MBConv3 layer
            data=self.MBConv4.output(data,self.km) # apply the MBConv4 layer
            data=self.MBConv5.output(data,self.km) # apply the MBConv5 layer
            data=self.MBConv6.output(data,self.km) # apply the MBConv6 layer
            data=self.MBConv7.output(data,self.km) # apply the MBConv7 layer
            data=self.conv1x1.output(data) # apply the 1x1 convolution layer with strides 1 and same padding
            if self.include_top:
                data=tf.reduce_mean(data,[1,2]) # apply global average pooling to get the mean value of each channel
                output=tf.nn.softmax(self.dense.output(data)) # apply the dense layer and softmax activation function to get the probability distribution of each class 
            else:
                if self.pooling=="avg":
                    data=tf.reduce_mean(data,[1,2])
                elif self.pooling=="max":
                    data=tf.reduce_max(data,[1,2])
        return output # return the output tensor
    
    
    def loss(self,output,labels,p):
        """A method that calculates the loss value between the output tensor and the labels tensor.

        Args:
            output: tensor, the output data after applying the model.
            labels: tensor, the true labels of the input data.
            p: integer, the index of the device to use.

        Returns:
            loss_value: tensor, the loss value.
        """
        with tf.device(assign_device(p,'GPU')): # assign the device to use
            loss_value=self.loss_object(labels,output) # calculate the loss value using categorical crossentropy loss function
        return loss_value # return the loss value
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,'GPU')):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
        return tape,output,loss
    
    
    def opt(self,gradient,p):
        """A method that updates the model parameters using the optimizer and the gradient.

        Args:
            gradient: tensor, the gradient of the loss function with respect to the model parameters.
            p: integer, the index of the device to use.

        Returns:
            param: list, the updated model parameters.
        """
        with tf.device(assign_device(p,'GPU')): # assign the device to use
            param=self.optimizer.opt(gradient,self.param,self.bc[0]) # update the model parameters using Adam optimizer and batch count
            return param # return the updated model parameters