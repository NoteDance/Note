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
        

def swish(x,beta=1.0):
    sigmoid=tf.math.sigmoid(beta * x)
    swish=x*sigmoid
    return swish


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
  'swish':swish,
  'softmax':tf.nn.softmax,
  # add more activation functions as needed
}
