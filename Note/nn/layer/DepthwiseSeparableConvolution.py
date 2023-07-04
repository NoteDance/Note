import tensorflow as tf
from Note.nn.initializer import initializer

class DepthwiseSeparableConvolution:
  """A class for implementing depthwise separable convolution for Reformer models."""

  def __init__(self, d_model, num_filters, filter_size, padding="SAME", weight_initializer='Xavier', bias_initializer='zeros', dtype='float32'):
    """Initializes the depthwise separable convolution layer.

    Args:
      d_model: int, the dimension of the model embeddings.
      num_filters: int, the number of filters for the pointwise convolution.
      filter_size: int, the size of the filter for the depthwise convolution.
      padding: str, the padding mode for the convolution, either "SAME" or "VALID".
    """
    self.d_model = d_model
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.padding = padding

    # Create the learnable parameters for the depthwise convolution
    self.depthwise_filter = initializer((1, filter_size, d_model, 1), weight_initializer, dtype)
    self.depthwise_bias = initializer((d_model,), bias_initializer, dtype)

    # Create the learnable parameters for the pointwise convolution
    self.pointwise_filter = initializer((1, 1, d_model, num_filters), weight_initializer, dtype)
    self.pointwise_bias = initializer((num_filters,), bias_initializer, dtype)
    self.param=[self.depthwise_filter, self.depthwise_bias, self.pointwise_filter, self.pointwise_bias]

  def output(self, data):
    """Applies the depthwise separable convolution layer on the input tensor.

    Args:
      data: tf.Tensor of shape [batch_size, seq_length, d_model], the input tensor.

    Returns:
      tf.Tensor of shape [batch_size, seq_length, num_filters], the output tensor after depthwise separable convolution.
    """
    # Reshape the input tensor to match the expected shape of [batch_size, height, width, channels]
    data = tf.expand_dims(data, axis=1)

    # Apply the depthwise convolution on the input tensor
    output = tf.nn.depthwise_conv2d(
        input=data,
        filter=self.depthwise_filter,
        strides=[1, 1, 1, 1],
        padding=self.padding,
        name="depthwise_conv2d"
    )

    # Add the depthwise bias to the output tensor
    output = output + self.depthwise_bias

    # Apply a non-linear activation function on the output tensor
    output = tf.nn.relu(output)

    # Apply the pointwise convolution on the output tensor
    output = tf.nn.conv2d(
        input=output,
        filters=self.pointwise_filter,
        strides=[1, 1, 1, 1],
        padding=self.padding,
        name="pointwise_conv2d"
    )

    # Add the pointwise bias to the output tensor
    output = output + self.pointwise_bias

    # Apply a non-linear activation function on the output tensor
    output = tf.nn.relu(output)

    # Reshape the output tensor to match the expected shape of [batch_size, seq_length, num_filters]
    output = tf.squeeze(output, axis=1)

    return output
