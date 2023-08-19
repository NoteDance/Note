import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict


class MBConv:
    """A class that implements the MBConv block, which is a mobile inverted residual block with squeeze-and-excitation optimization.

    Args:
        input_size: integer, the number of input channels.
        output_size: integer, the number of output channels.
        kernel_size: integer, the dimension of the convolution window. Default is 3.
        strides: integer, the stride of the convolution. Default is 1.
        expand_ratio: integer, scaling coefficient for the input channels. Default is 1.
        repeats: integer, how many times to repeat the same MBConv block. Default is 1.
        dtype: string, the data type of the variables. Default is 'float32'.
    """
    def __init__(self, input_size, output_size, kernel_size=3, strides=1, expand_ratio=1, repeats=1, dtype='float32'):
        self.expanded_size = input_size * expand_ratio # calculate the expanded size by multiplying the input size by the expand ratio
        self.output_size = output_size # store the output size
        self.weight_depthwise = initializer([kernel_size, kernel_size, self.expanded_size, 1], 'Xavier', dtype) # initialize the depthwise convolution kernel with Xavier initialization
        self.weight_project = initializer([1, 1, self.expanded_size, output_size], 'Xavier', dtype) # initialize the projection convolution kernel with Xavier initialization
        self.weight_se_1 = initializer([1, self.expanded_size // 4, self.expanded_size], 'Xavier', dtype) # initialize the first squeeze-and-excitation weight matrix with Xavier initialization
        self.weight_se_2 = initializer([1, self.expanded_size, self.expanded_size], 'Xavier', dtype) # initialize the second squeeze-and-excitation weight matrix with Xavier initialization
        self.strides = strides # store the strides
        self.expand_ratio = expand_ratio # store the expand ratio
        self.repeats = repeats # store the repeats
        self.swish = activation_dict['swish'] # get the swish activation function from the activation dictionary
        self.dtype=dtype # store the data type
        self.param = [self.weight_depthwise, self.weight_project, self.weight_se_1, self.weight_se_2] # store all the parameters in a list


    def output(self, data, train_flag=True):
        """A method that performs forward propagation on the input data and returns the output tensor.

        Args:
            data: tensor, the input data.

        Returns:
            output: tensor, the output data after applying MBConv block(s).
        """
        for i in range(self.repeats): # loop over each repeat
            if i == 0: # if it is the first repeat
                strides_i = self.strides # use the given strides
                inputs_i = data # use the original input data
            else: # if it is not the first repeat
                strides_i = 1 # use unit strides
                inputs_i = x # use the previous output data as input data
            if self.expand_ratio > 1: # if the expand ratio is larger than 1
                weight_expand = initializer([1, 1, inputs_i.shape[-1], self.expanded_size], 'Xavier', self.dtype) # dynamically create a temporary weight_expand variable according to the input shape
                x = tf.nn.conv2d(inputs_i ,weight_expand ,strides=[1 ,1 ,1 ,1] ,padding="SAME") # apply a 1x1 convolution to expand the input channels
                if train_flag == True:
                    x = tf.nn.batch_normalization(x ,tf.Variable(tf.zeros([self.expanded_size])) ,tf.Variable(tf.ones([self.expanded_size])) ,None ,None ,1e-5) # apply batch normalization to normalize the output
                x = self.swish(x) # apply swish activation function to increase nonlinearity
            else: # if the expand ratio is not larger than 1
                x = inputs_i # skip the expansion phase and use the input data directly
            x = tf.nn.depthwise_conv2d(x ,self.weight_depthwise ,strides=[1 ,strides_i ,strides_i ,1] ,padding="SAME") # apply a depthwise convolution to process each channel separately
            if train_flag == True:
                x = tf.nn.batch_normalization(x ,tf.Variable(tf.zeros([self.expanded_size])) ,tf.Variable(tf.ones([self.expanded_size])) ,None ,None ,1e-5) # apply batch normalization to normalize the output
            x = self.swish(x) # apply swish activation function to increase nonlinearity
            se_tensor = tf.reduce_mean(x ,[1 ,2]) # global average pooling to get the mean value of each channel
            se_tensor = tf.reshape(se_tensor ,[1 ,1 ,-1]) # reshape it to facilitate subsequent operations
            se_tensor = tf.nn.conv1d(se_tensor ,self.weight_se_1 ,stride=1 ,padding="SAME") # use a fully connected layer to reduce the channel number to a quarter of the original size
            se_tensor = self.swish(se_tensor) # use swish activation function to increase nonlinearity
            se_tensor = tf.nn.conv1d(se_tensor ,self.weight_se_2 ,stride=1 ,padding="SAME") # use another fully connected layer to restore the channel number to the original size
            se_tensor = tf.nn.sigmoid(se_tensor) # use sigmoid activation function to get the weight coefficient of each channel, ranging from 0 to 1
            x = tf.multiply(x, se_tensor) # multiply the output tensor by the squeeze-and-excitation tensor element-wise
            x = tf.nn.conv2d(x, self.weight_project, strides=[1, 1, 1, 1], padding="SAME") # apply a 1x1 convolution to project the output channels to the desired size
            if train_flag == True:
                x = tf.nn.batch_normalization(x, tf.Variable(tf.zeros([self.output_size])), tf.Variable(tf.ones([self.output_size])), None, None, 1e-5) # apply batch normalization to normalize the output
            if inputs_i.shape == x.shape: # if the input shape and the output shape are the same
                x = tf.add(x, inputs_i) # add the input tensor and the output tensor element-wise to form a residual connection
            if train_flag == True:
                x = tf.nn.dropout(x, rate=0.2) # apply dropout to prevent overfitting
        return x # return the final output tensor after applying MBConv block(s)
