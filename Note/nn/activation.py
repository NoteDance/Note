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
        elif activation_func=='softmax':
            return tf.nn.softmax(tf.matmul(data,weight)+bias)
        elif activation_func=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight)+bias)
        elif activation_func=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight)+bias)
        elif type(activation_func)==list and activation_func[0]=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight)+bias,activation_func[1])
        elif type(activation_func)==list and activation_func[0]=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight)+bias,activation_func[1])
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
        elif activation_func=='softmax':
            return tf.nn.softmax(tf.matmul(data,weight))
        elif activation_func=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight))
        elif activation_func=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight))
        elif type(activation_func)==list and activation_func[0]=='leaky_relu':
            return tf.nn.leaky_relu(tf.matmul(data,weight),activation_func[1])
        elif type(activation_func)==list and activation_func[0]=='gelu':
            return tf.nn.gelu(tf.matmul(data,weight),activation_func[1])
        else:
            return tf.matmul(data,weight)


def activation_conv(data,weight,activation_func,strides,padding,data_format,dilations,conv_func,bias=None):
    if bias is None:
        if activation=='sigmoid':
            return tf.nn.sigmoid(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='tanh':
            return tf.nn.tanh(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='relu':
            return tf.nn.relu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='elu':
            return tf.nn.elu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation=='softmax':
            return tf.nn.softmax(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation_func=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif activation_func=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations))
        elif type(activation_func)==list and activation_func[0]=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations),activation_func[1])
        elif type(activation_func)==list and activation_func[0]=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations),activation_func[1])
        else:
            return conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)
    else:
        if activation=='sigmoid':
            return tf.nn.sigmoid(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='tanh':
            return tf.nn.tanh(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='relu':
            return tf.nn.relu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='elu':
            return tf.nn.elu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation=='softmax':
            return tf.nn.softmax(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation_func=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif activation_func=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias)
        elif type(activation_func)==list and activation_func[0]=='leaky_relu':
            return tf.nn.leaky_relu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias,activation_func[1])
        elif type(activation_func)==list and activation_func[0]=='gelu':
            return tf.nn.gelu(conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias,activation_func[1])
        else:
            return conv_func(data,weight,strides=strides,padding=padding,data_format=data_format,dilations=dilations)+bias
