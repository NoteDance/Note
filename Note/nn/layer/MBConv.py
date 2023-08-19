import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict


class MBConv:
    def __init__(self, input_size, output_size, kernel_size=3, strides=1, expand_ratio=1, repeats=1, se_ratio=0.25, model_number=0, dtype='float32'):
        self.expanded_size = input_size * expand_ratio # compute the expanded size of the input channels
        self.output_size = output_size # store the output size of the output channels
        self.weight_depthwise = initializer([kernel_size, kernel_size, self.expanded_size, 1], 'Xavier', dtype) # create a weight tensor for the depthwise convolution
        self.weight_project = initializer([1, 1, self.expanded_size, output_size], 'Xavier', dtype) # create a weight tensor for the projection convolution
        self.weight_se_1 = initializer([1, 1, self.expanded_size, int(self.expanded_size * se_ratio)], 'Xavier', dtype) # create a weight tensor for the first squeeze and excitation convolution
        self.weight_se_2 = initializer([1, 1, int(self.expanded_size * se_ratio), self.expanded_size], 'Xavier', dtype) # create a weight tensor for the second squeeze and excitation convolution
        self.strides = strides # store the strides for the depthwise convolution
        self.expand_ratio = expand_ratio # store the expand ratio for the expansion step
        self.repeats = repeats # store the number of repeats for the MBConv module
        self.se_ratio = se_ratio # store the se ratio for the squeeze and excitation module
        self.model_number = model_number # store the model number for computing the drop connect probability
        self.dtype=dtype # store the data type for the tensors
        self.swish = activation_dict['swish'] # get the swish activation function from the activation dictionary
        self.param = [self.weight_depthwise, self.weight_project, self.weight_se_1, self.weight_se_2] # store all the weight tensors in a list


    def output(self, data, train_flag=True):
        for i in range(self.repeats): # loop over the number of repeats
            if i == 0: # if it is the first repeat
                strides_i = self.strides # use the given strides
                inputs_i = data # use the input data as inputs_i
            else: # if it is not the first repeat
                strides_i = 1 # use strides 1
                inputs_i = x # use x as inputs_i
            if self.expand_ratio > 1: # if expand ratio is larger than 1
                weight_expand = initializer([1, 1, inputs_i.shape[-1], self.expanded_size], 'Xavier', self.dtype) # create a weight tensor for the expansion convolution
                x = tf.nn.conv2d(inputs_i ,weight_expand ,strides=[1 ,1 ,1 ,1] ,padding="SAME") # apply a 1x1 convolution to expand the input channels to expanded size
                if train_flag: # if it is in training mode
                    x = tf.nn.batch_normalization(x ,tf.Variable(tf.zeros([self.expanded_size])) ,tf.Variable(tf.ones([self.expanded_size])) ,None ,None ,1e-5) # apply batch normalization to normalize the output
                x = self.swish(x) # apply swish activation function to increase nonlinearity
            else: # if expand ratio is not larger than 1
                x = inputs_i # use inputs_i as x
            x = tf.nn.depthwise_conv2d(x ,self.weight_depthwise ,strides=[1 ,strides_i ,strides_i ,1] ,padding="SAME") # apply a depthwise convolution to process each channel separately
            if train_flag: # if it is in training mode
                x = tf.nn.batch_normalization(x ,tf.Variable(tf.zeros([self.expanded_size])) ,tf.Variable(tf.ones([self.expanded_size])) ,None ,None ,1e-5) # apply batch normalization to normalize the output
            x = self.swish(x) # apply swish activation function to increase nonlinearity
            if 0 < self.se_ratio <= 1: # if se ratio is positive and not larger than 1
                se_tensor = tf.reduce_mean(x, axis=[1, 2]) # compute the global average pooling of each channel
                se_tensor = tf.reshape(se_tensor, [-1, 1, 1, self.expanded_size]) # reshape it to match the shape of x
                se_tensor = tf.nn.conv2d(se_tensor, self.weight_se_1, strides=[1, 1, 1, 1], padding="SAME") # use a 1x1 convolution to reduce the channel number to a fraction of the original size, controlled by se ratio
                se_tensor = self.swish(se_tensor) # use swish activation function to increase nonlinearity
                se_tensor = tf.nn.conv2d(se_tensor, self.weight_se_2, strides=[1, 1, 1, 1], padding="SAME") # use another 1x1 convolution to restore the channel number to the original size
                se_tensor = tf.nn.sigmoid(se_tensor) # use sigmoid activation function to get the weight coefficient of each channel, ranging from 0 to 1
                x = tf.multiply(x, se_tensor) # multiply the output tensor by the squeeze and excitation tensor element-wise
            x = tf.nn.conv2d(x, self.weight_project, strides=[1, 1, 1, 1], padding="SAME") # apply a 1x1 convolution to project the output channels to the desired size
            x = tf.nn.batch_normalization(x, tf.Variable(tf.zeros([self.output_size])), tf.Variable(tf.ones([self.output_size])), None, None, 1e-5) # apply batch normalization to normalize the output
            if inputs_i.shape == x.shape: # if the input shape and the output shape are the same
                if train_flag: # if it is in training mode
                    rate = 0.2 * (1 - 0.5 * (self.model_number + 1) / 7) # compute the drop connect probability according to the paper formula, where i is the model number (from 0 to 6)
                    x = tf.nn.dropout(x, rate=rate, noise_shape=(None, 1, 1, 1)) # apply drop connect to randomly drop some residual connections
                x = tf.add(x, inputs_i) # add the input tensor and the output tensor element-wise to form a residual connection
        return x # return the output tensor
