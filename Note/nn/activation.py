import tensorflow as tf


def activation(data,weight,bias=None,activation=None,use_bias=True):
    if use_bias==True:
        if activation=='sigmoid':
            return tf.nn.sigmoid(tf.matmul(data,weight)+bias)
        elif activation=='tanh':
            return tf.nn.tanh(tf.matmul(data,weight)+bias)
        elif activation=='relu':
            return tf.nn.relu(tf.matmul(data,weight)+bias)
        elif activation=='elu':
            return tf.nn.elu(tf.matmul(data,weight)+bias)
        elif activation=='softmax':
            return tf.nn.softmax(tf.matmul(data,weight)+bias)
        elif activation=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight)+bias)
        elif activation=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight)+bias)
        elif type(activation)==list and activation[0]=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight)+bias,activation[1])
        elif type(activation)==list and activation[0]=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight)+bias,activation[1])
        else:
            return tf.matmul(data,weight)+bias
    else:
        if activation=='sigmoid':
            return tf.nn.sigmoid(tf.matmul(data,weight))
        elif activation=='tanh':
            return tf.nn.tanh(tf.matmul(data,weight))
        elif activation=='relu':
            return tf.nn.relu(tf.matmul(data,weight))
        elif activation=='elu':
            return tf.nn.elu(tf.matmul(data,weight))
        elif activation=='softmax':
            return tf.nn.softmax(tf.matmul(data,weight))
        elif activation=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight))
        elif activation=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight))
        elif type(activation)==list and activation[0]=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight),activation[1])
        elif type(activation)==list and activation[0]=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight),activation[1])
        else:
            return tf.matmul(data,weight)


def activation_conv(data,weight,activation,strides,padding,data_format,dilations,conv_func,bias=None):
    if bias is None:
        if activation=='sigmoid':
            return tf.nn.sigmoid(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='tanh':
            return tf.nn.tanh(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='relu':
            return tf.nn.relu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='elu':
            return tf.nn.elu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='softmax':
            return tf.nn.softmax(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif type(activation)==list and activation[0]=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations),activation[1])
        elif type(activation)==list and activation[0]=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations),activation[1])
        else:
            return conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)
    else:
        if activation=='sigmoid':
            return tf.nn.sigmoid(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='tanh':
            return tf.nn.tanh(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='relu':
            return tf.nn.relu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='elu':
            return tf.nn.elu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='softmax':
            return tf.nn.softmax(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif type(activation)==list and activation[0]=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias,activation[1])
        elif type(activation)==list and activation[0]=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias,activation[1])
        else:
            return conv_func(data,weight,strides,padding=padding,data_format=data_format,dilations=dilations)+bias


def activation_conv_transpose(data,weight,output_shape,activation,strides,padding,data_format,dilations,conv_func,bias=None):
    if bias is None:
        if activation=='sigmoid':
            return tf.nn.sigmoid(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='tanh':
            return tf.nn.tanh(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='relu':
            return tf.nn.relu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='elu':
            return tf.nn.elu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='softmax':
            return tf.nn.softmax(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='gelu':
            return tf.nn.gelu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations))
        elif type(activation)==list and activation[0]=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations),activation[1])
        elif type(activation)==list and activation[0]=='gelu':
            return tf.nn.gelu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations),activation[1])
        else:
            return conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)
    else:
        if activation=='sigmoid':
            return tf.nn.sigmoid(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='tanh':
            return tf.nn.tanh(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='relu':
            return tf.nn.relu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='elu':
            return tf.nn.elu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='softmax':
            return tf.nn.softmax(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='gelu':
            return tf.nn.gelu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif type(activation)==list and activation[0]=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias,activation[1])
        elif type(activation)==list and activation[0]=='gelu':
            return tf.nn.gelu(conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias,activation[1])
        else:
            return conv_func(data,weight,output_shape,strides,padding=padding,data_format=data_format,dilations=dilations)+bias


def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.

    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    Args:
        x: A tensor or variable.

    Returns:
        A tensor.
    """
    point_two = tf.constant(0.2, x.dtype)
    point_five = tf.constant(0.5, x.dtype)
    x = tf.multiply(x, point_two)
    x = tf.add(x, point_five)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x


def mish(x):
    """Mish activation function.

    It is defined as:

    ```python
    def mish(x):
        return x * tanh(softplus(x))
    ```

    where `softplus` is defined as:

    ```python
    def softplus(x):
        return log(exp(x) + 1)
    ```

    Args:
        x: Input tensor.

    Returns:
        The mish activation.

    Reference:
        - [Mish: A Self Regularized Non-Monotonic
        Activation Function](https://arxiv.org/abs/1908.08681)
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def glu(x, axis=-1):
    a, b = tf.split(x, 2, axis=axis)
    gate = tf.nn.sigmoid(b)
    out = tf.math.multiply(a, gate)
    return out


def celu(x, alpha=1.0):
    out = tf.math.maximum(0.0, x) + tf.math.minimum(0.0, alpha * (tf.math.exp(x / alpha) - 1.0))
    return out


def logsigmoid(x):
  return tf.math.log(1 + tf.math.exp(-x))


def tanhshrink(x):
  return x - tf.math.tanh(x)


def hardswish(x):
    return x * tf.nn.relu6(x + 3) / 6


activation_dict={
  'tanh':tf.nn.tanh,
  'relu':tf.nn.relu,
  'sigmoid':tf.nn.sigmoid,
  'elu':tf.nn.elu,
  'leaky_relu':tf.nn.leaky_relu,
  'gelu':tf.nn.gelu,
  'crelu':tf.nn.crelu,
  'relu6':tf.nn.relu6,
  'selu':tf.nn.selu,
  'silu':tf.nn.silu,
  'swish':tf.nn.silu,
  'softmax':tf.nn.softmax,
  'softsign':tf.nn.softsign,
  'hard_sigmoid':hard_sigmoid,
  'softplus':tf.math.softplus,
  'mish':mish,
  'glu':glu,
  'celu':celu,
  'logsigmoid':logsigmoid,
  'tanhshrink':tanhshrink,
  'hardswish':hardswish
}
