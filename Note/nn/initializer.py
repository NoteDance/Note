import tensorflow as tf


def initializer(shape,initializer,dtype):
    if type(initializer)==list:
        if initializer[0]=='normal':
           param=tf.Variable(tf.random.normal(shape=shape,mean=initializer[1],stddev=initializer[2],dtype=dtype))
        elif initializer[0]=='uniform':
            param=tf.Variable(tf.random.uniform(shape=shape,minval=initializer[1],maxval=initializer[2],dtype=dtype))
    else:
        if initializer=='normal':
            param=tf.Variable(tf.random.normal(shape=shape,dtype=dtype))
        elif initializer=='uniform':
            param=tf.Variable(tf.random.uniform(shape=shape,maxval=0.01,dtype=dtype))
        elif initializer=='zero':
            param=tf.Variable(tf.zeros(shape=shape,dtype=dtype))
        elif initializer=='xavier':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.sqrt(2.0/(fan_in+fan_out))
            param=tf.Variable(tf.random.normal(shape=shape,mean=0.0,stddev=scale,dtype=dtype))
    return param