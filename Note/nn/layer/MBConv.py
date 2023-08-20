import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict


# Define a class named MBConv, which implements a modified depthwise separable convolution (Mobile Inverted Residual Bottleneck)
class MBConv:
    # Define the initialization method, which takes several parameters to create the convolution layers
    def __init__(self, input_size, output_size, kernel_size=3, strides=1, expand_ratio=1, repeats=1, se_ratio=0.25, model_number=0, dtype='float32'):
        # Calculate the expanded size based on the input size and the expand ratio
        self.expanded_size = input_size * expand_ratio
        # Assign the output size to an attribute
        self.output_size = output_size
        # Initialize empty lists to store the weight matrices for each layer
        self.weight_expand = []
        self.weight_depthwise = []
        self.weight_project = []
        self.weight_se_1 = []
        self.weight_se_2 = []
        # Calculate the number of channels for the SE module based on the expanded size and the SE ratio
        se_channels = max(1, int(self.expanded_size * se_ratio))
        # Loop through the number of repeats for each MBConv block
        for i in range(repeats):
            # If it is the first block, use the input size as the input channels
            if i==0:
                self.weight_expand.append(initializer([1, 1, input_size, self.expanded_size], 'Xavier', dtype))
                self.weight_depthwise.append(initializer([kernel_size, kernel_size, self.expanded_size, 1], 'Xavier', dtype))
                self.weight_project.append(initializer([1, 1, self.expanded_size, output_size], 'Xavier', dtype))
            # Otherwise, use the output size as the input channels
            else:
                self.weight_expand.append(initializer([1, 1, output_size, self.expanded_size], 'Xavier', dtype))
                self.weight_depthwise.append(initializer([kernel_size, kernel_size, self.expanded_size, 1], 'Xavier', dtype))
                self.weight_project.append(initializer([1, 1, self.expanded_size, output_size], 'Xavier', dtype))
            # Initialize the weight matrices for the SE module
            self.weight_se_1.append(initializer([1, 1, self.expanded_size, se_channels], 'Xavier', dtype))
            self.weight_se_2.append(initializer([1, 1, se_channels, self.expanded_size], 'Xavier', dtype))
        # Assign the other parameters to attributes
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.repeats = repeats
        self.se_ratio = se_ratio
        self.model_number = model_number
        # Get the swish activation function from a dictionary of activation functions
        self.swish = activation_dict['swish']
        # Store all the weight matrices in a list of parameters
        self.param = [self.weight_expand, self.weight_depthwise, self.weight_project, self.weight_se_1, self.weight_se_2]
    
    
    # Define the output method, which takes an input tensor and a training flag to compute the output tensor of the MBConv block
    def output(self, data, train_flag=True):
        # Loop through the number of repeats for each MBConv block
        for i in range(self.repeats):
            # If it is the first block, use the strides parameter as the strides for the depthwise convolution layer
            if i == 0:
                strides_i = self.strides
                inputs_i = data
            # Otherwise, use 1 as the strides for the depthwise convolution layer
            else:
                strides_i = 1
                inputs_i = x
            # If the expand ratio is greater than 1, apply an expansion layer to increase the number of channels
            if self.expand_ratio != 1:
                x = tf.nn.conv2d(inputs_i ,self.weight_expand[i] ,strides=[1 ,1 ,1 ,1] ,padding="SAME")
                if train_flag:
                    x = tf.nn.batch_normalization(x ,tf.Variable(tf.zeros([self.expanded_size])) ,tf.Variable(tf.ones([self.expanded_size])) ,None ,None ,1e-5)
                x = self.swish(x)
            # Otherwise, skip the expansion layer and use the input tensor as it is
            else:
                x = inputs_i
            # Apply a depthwise convolution layer to apply spatial filters to each channel
            x = tf.nn.depthwise_conv2d(x ,self.weight_depthwise[i] ,strides=[1 ,strides_i ,strides_i ,1] ,padding="SAME")
            if train_flag:
                x = tf.nn.batch_normalization(x ,tf.Variable(tf.zeros([self.expanded_size])) ,tf.Variable(tf.ones([self.expanded_size])) ,None ,None ,1e-5)
            x = self.swish(x)
            # If the SE ratio is positive and less than or equal to 1, apply an SE module to squeeze and excite the channels
            if 0 < self.se_ratio <= 1:
                # Compute the global average pooling of the input tensor
                se_tensor = tf.reduce_mean(x, axis=[1, 2])
                # Reshape the pooled tensor to match the number of channels
                se_tensor = tf.reshape(se_tensor, [-1, 1, 1, self.expanded_size])
                # Apply a pointwise convolution layer to reduce the number of channels
                se_tensor = tf.nn.conv2d(se_tensor, self.weight_se_1[i], strides=[1, 1, 1, 1], padding="SAME")
                # Apply the swish activation function
                se_tensor = self.swish(se_tensor)
                # Apply another pointwise convolution layer to restore the number of channels
                se_tensor = tf.nn.conv2d(se_tensor, self.weight_se_2[i], strides=[1, 1, 1, 1], padding="SAME")
                # Apply the sigmoid activation function
                se_tensor = tf.nn.sigmoid(se_tensor)
                # Multiply the input tensor and the SE tensor element-wise
                x = tf.multiply(x, se_tensor)
            # Apply a projection layer to reduce the number of channels
            x = tf.nn.conv2d(x, self.weight_project[i], strides=[1, 1, 1, 1], padding="SAME")
            if train_flag:
                x = tf.nn.batch_normalization(x, tf.Variable(tf.zeros([self.output_size])), tf.Variable(tf.ones([self.output_size])), None, None, 1e-5)
            # If the input tensor and the output tensor have the same shape, add them together to form a residual connection
            if inputs_i.shape == x.shape:
                if train_flag:
                    # Compute the dropout rate based on the model number
                    rate = 0.2 * (1 - 0.5 * (self.model_number + 1) / 7)
                    # Apply dropout to the output tensor
                    x = tf.nn.dropout(x, rate=rate, noise_shape=(None, 1, 1, 1))
                    # Add the input tensor and the output tensor element-wise to form a residual connection
                    x = tf.add(x, inputs_i)
        # Return the output tensor
        return x
