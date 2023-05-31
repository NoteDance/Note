import tensorflow as tf


def activation(data,weight,bias=None,activation_func=None,use_bias=True):
    if use_bias==True:
        if activation_func=='sigmoid':
            return tf.nn.sigmoid(tf.matmul(data,weight)+bias)
        elif activation_func=='tanh':
            return tf.nn.tanh(tf.matmul(data,weight)+bias)
        elif activation_func=='relu':
            return tf.nn.relu(tf.matmul(data,weight)+bias)
        elif activation_func=='elu':
            return tf.nn.elu(tf.matmul(data,weight)+bias)
        else:
            return tf.matmul(data,weight)+bias
    else:
        if activation_func=='sigmoid':
            return tf.nn.sigmoid(tf.matmul(data,weight))
        elif activation_func=='tanh':
            return tf.nn.tanh(tf.matmul(data,weight))
        elif activation_func=='relu':
            return tf.nn.relu(tf.matmul(data,weight))
        elif activation_func=='elu':
            return tf.nn.elu(tf.matmul(data,weight))
        else:
            return tf.matmul(data,weight)


def activation_conv(data,weight,activation_func,strides,padding,data_format,dilations,conv_func):
    if activation=='sigmoid':
        return tf.nn.sigmoid(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
    elif activation=='tanh':
        return tf.nn.tanh(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
    elif activation=='relu':
        return tf.nn.relu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
    elif activation=='elu':
        return tf.nn.elu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
    else:
        return conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)