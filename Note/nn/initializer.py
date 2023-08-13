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
        elif initializer=='zeros':
            param=tf.Variable(tf.zeros(shape=shape,dtype=dtype))
        elif initializer=='Xavier':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.sqrt(2.0/(fan_in+fan_out))
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-scale*tf.sqrt(3.0),maxval=scale*tf.sqrt(3.0),dtype=dtype))
        elif initializer=='Xavier_normal':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.sqrt(2.0/(fan_in+fan_out))
            param=tf.Variable(tf.random.normal(shape=shape,mean=0.0,stddev=scale,dtype=dtype))
        elif initializer=='He':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.sqrt(2.0/fan_in)
            param=tf.Variable(tf.random.normal(shape=shape,mean=0.0,stddev=scale,dtype=dtype))
        elif initializer=='He_uniform':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.sqrt(2.0/fan_in)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-scale*tf.sqrt(3.0),maxval=scale*tf.sqrt(3.0),dtype=dtype))
        elif initializer=='Lecun':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.sqrt(3.0/fan_in)
            param=tf.Variable(tf.random.normal(shape=shape,mean=0.0,stddev=scale,dtype=dtype))
        elif initializer=='Lecun_uniform':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.sqrt(3.0/fan_in)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-scale*tf.sqrt(3.0),maxval=scale*tf.sqrt(3.0),dtype=dtype))
    return param
