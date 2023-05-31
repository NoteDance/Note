import tensorflow as tf


def initializer(shape,initializer,dtype):
    if type(initializer)==list:
        if initializer[0]=='normal':
           param=tf.Variable(tf.random.normal(shape=shape,mean=initializer[1],stddev=initializer[2],dtype=dtype))
        elif initializer[0]=='uniform':
            param=tf.Variable(tf.random.uniform(shape=shape,minval=initializer[1],maxval=initializer[2],dtype=dtype))
    else:
        if initializer=='normal':
            if dtype!=None:
                param=tf.Variable(tf.random.normal(shape=shape,dtype=dtype))
            else:
                param=tf.Variable(tf.random.normal(shape=shape))
        elif initializer=='uniform':
            if dtype!=None:
                param=tf.Variable(tf.random.uniform(shape=shape,maxval=0.01,dtype=dtype))
            else:
                param=tf.Variable(tf.random.uniform(shape=shape,maxval=0.01))
        elif initializer=='zero':
            if dtype!=None:
                param=tf.Variable(tf.zeros(shape=shape,dtype=dtype))
            else:
                param=tf.Variable(tf.zeros(shape=shape))
    return param