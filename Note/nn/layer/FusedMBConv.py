import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict


# Define a class for the FusedMBConv block
class FusedMBConv:
    # Initialize the class with the input and output parameters
    def __init__(self, input_size, output_size, kernel_size=3, strides=1, expand_ratio=1, repeats=1, se_ratio=0, rate=0.2, dtype='float32'):
        # Calculate the expanded size by multiplying the input size by the expand ratio
        self.expanded_size = input_size * expand_ratio
        # Assign the output size to a class attribute
        self.output_size = output_size
        # Initialize empty lists for storing the weights of the convolution layers
        self.weight_expand = []
        self.weight_project = []
        self.weight_se_1 = []
        self.weight_se_2 = []
        # Calculate the number of channels for the squeeze and excite layer
        se_channels = max(1, int(self.expanded_size * se_ratio))
        # Loop over the number of repeats for the block
        for i in range(repeats):
            # If it is the first repeat, use the input size as the input channels for the expand layer
            if i==0:
                # Initialize the weight for the expand layer using a custom initializer from Note.nn module
                self.weight_expand.append(initializer([kernel_size, kernel_size, input_size, self.expanded_size], 'Xavier', dtype))
                # Initialize the weight for the project layer using a custom initializer from Note.nn module
                self.weight_project.append(initializer([1 if expand_ratio != 1 else kernel_size, 1 if expand_ratio != 1 else kernel_size, self.expanded_size, output_size], 'Xavier', dtype))
            # Otherwise, use the output size as the input channels for the expand layer
            else:
                # Initialize the weight for the expand layer using a custom initializer from Note.nn module
                self.weight_expand.append(initializer([kernel_size, kernel_size, output_size, self.expanded_size], 'Xavier', dtype))
                # Initialize the weight for the project layer using a custom initializer from Note.nn module
                self.weight_project.append(initializer([1 if expand_ratio != 1 else kernel_size, 1 if expand_ratio != 1 else kernel_size, self.expanded_size, output_size], 'Xavier', dtype))
            # Initialize the weights for the squeeze and excite layers using a custom initializer from Note.nn module
            self.weight_se_1.append(initializer([1, 1, self.expanded_size, se_channels], 'Xavier', dtype))
            self.weight_se_2.append(initializer([1, 1, se_channels, self.expanded_size], 'Xavier', dtype))
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
        # Store all the weights in a list as a class attribute
        self.param = [self.weight_expand, self.weight_project, self.weight_se_1, self.weight_se_2]
    
    
    # Define a method for computing the output of the block given an input tensor and a training flag
    def output(self, data, train_flag=True):
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
            # If the expand ratio is not 1, apply an expansion step using a convolution layer with kernel size x kernel size filter and no bias term 
            if self.expand_ratio != 1:
                x = tf.nn.conv2d(inputs_i ,self.weight_expand[i] ,strides=strides_i ,padding="SAME")
                # If it is training mode, apply batch normalization to normalize the output tensor along its channel dimension 
                if train_flag:
                    x = tf.nn.batch_normalization(x ,tf.Variable(tf.zeros([self.expanded_size])) ,tf.Variable(tf.ones([self.expanded_size])) ,None ,None ,1e-5)
                # Apply swish activation function to the output tensor
                x = self.swish(x)
            # Otherwise, skip the expansion step and use the input tensor as the output tensor
            else:
                x = inputs_i
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
            # Apply a projection step using a convolution layer with 1x1 filter (or kernel size x kernel size filter if expand ratio is 1) and no bias term to reduce the number of channels to output_size
            x = tf.nn.conv2d(x, self.weight_project[i], strides=1 if self.expand_ratio != 1 else strides_i, padding="SAME")
            # If it is training mode, apply batch normalization to normalize the output tensor along its channel dimension 
            if train_flag:
                x = tf.nn.batch_normalization(x, tf.Variable(tf.zeros([self.output_size])), tf.Variable(tf.ones([self.output_size])), None, None, 1e-5)
            # If the input tensor and the output tensor have the same shape, apply a residual connection by adding them element-wise
            if inputs_i.shape == x.shape:
                # If it is training mode, apply dropout to the output tensor with a variable rate that depends on b and the repeats parameter and a noise shape
                if train_flag:
                    rate = self.rate * self.b / self.repeats
                    x = tf.nn.dropout(x, rate=rate, noise_shape=(None, 1, 1, 1))
                    x = tf.add(x, inputs_i)
            # Increment b by 1 for the next iteration
            self.b.assign_add(1)
        # Return the output tensor
        return x
