import tensorflow as tf
from Note import nn


def conv2d_func(input, weight, bias=None, strides=1, padding=0, dilations=1, groups=1):
    if not isinstance(padding,str):
        x = nn.zeropadding2d(padding=padding)(input)
        padding = 'VALID'
    if groups == 1:
        if bias:
            x = tf.nn.conv2d(x, weight, strides, padding, dilations=dilations) + bias
        else:
            x = tf.nn.conv2d(x, weight, strides, padding, dilations=dilations)
    else:
        input_groups = tf.split(input, num_or_size_splits=groups, axis=-1)
        weight_groups = tf.split(weight, num_or_size_splits=groups, axis=-1)
        output_groups = []
        for i in range(groups):
            x = tf.nn.conv2d(input_groups[i], weight_groups[i], strides, padding, dilations=dilations)
            output_groups.append(x)
        x = tf.concat(output_groups, axis=-1)
        if bias:
            x = x + bias
    return x