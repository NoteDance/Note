import tensorflow as tf


def initializer(shape,initializer,dtype):
    if type(initializer)==list:
        if initializer[0]=='normal':
            param=tf.Variable(tf.random.normal(shape=shape,mean=initializer[1],stddev=initializer[2],dtype=dtype))
        elif initializer[0]=='uniform':
            param=tf.Variable(tf.random.uniform(shape=shape,minval=initializer[1],maxval=initializer[2],dtype=dtype))
        elif initializer[0]=='VarianceScaling':
            fan_in = tf.reduce_prod(shape[:-1])
            fan_out = shape[-1]
            if initializer[2] == 'fan_in':
                n = fan_in
            elif initializer[2] == 'fan_out':
                n = fan_out
            elif initializer[2] == 'fan_avg':
                n = (fan_in + fan_out) / 2.0
            else:
                raise ValueError('Invalid mode: ' + initializer[2])
            if initializer[3] == 'truncated_normal':
                stddev = tf.sqrt(initializer[1] / n)
                param=tf.Variable(tf.random.truncated_normal(shape=shape, mean=0.0, stddev=stddev, dtype=dtype))
            elif initializer[3] == 'uniform':
                limit = tf.sqrt(3.0 * initializer[1] / n)
                param=tf.Variable(tf.random.uniform(shape=shape, minval=-limit, maxval=limit, dtype=dtype))
            else:
                raise ValueError('Invalid distribution: ' + initializer[3])
        elif initializer[0]=='truncated_normal':
            param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=initializer[1],dtype=dtype))
    else:
        if initializer=='normal':
            param=tf.Variable(tf.random.normal(shape=shape,dtype=dtype))
        elif initializer=='uniform':
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-0.05,maxval=0.05,dtype=dtype))
        elif initializer=='zeros':
            param=tf.Variable(tf.zeros(shape=shape,dtype=dtype))
        elif initializer=='ones':
            param=tf.Variable(tf.ones(shape=shape,dtype=dtype))
        elif initializer=='Xavier':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.cast(tf.sqrt(2.0/(fan_in+fan_out)),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-scale*tf.cast(tf.sqrt(3.0),dtype),maxval=scale*tf.cast(tf.sqrt(3.0),dtype),dtype=dtype))
        elif initializer=='Xavier_normal':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.cast(tf.sqrt(2.0/(fan_in+fan_out)),dtype)
            param=tf.Variable(tf.random.normal(shape=shape,mean=0.0,stddev=scale,dtype=dtype))
        elif initializer=='He':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.cast(tf.sqrt(2.0/fan_in),dtype)
            param=tf.Variable(tf.random.normal(shape=shape,mean=0.0,stddev=scale,dtype=dtype))
        elif initializer=='He_uniform':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.cast(tf.sqrt(2.0/fan_in),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-scale*tf.cast(tf.sqrt(3.0),dtype),maxval=scale*tf.cast(tf.sqrt(3.0),dtype),dtype=dtype))
        elif initializer=='Lecun':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.cast(tf.sqrt(3.0/fan_in),dtype)
            param=tf.Variable(tf.random.normal(shape=shape,mean=0.0,stddev=scale,dtype=dtype))
        elif initializer=='Lecun_uniform':
            fan_in=shape[0]
            fan_out=shape[1] if len(shape)>1 else 1
            scale=tf.cast(tf.sqrt(3.0/fan_in),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-scale*tf.cast(tf.sqrt(3.0),dtype),maxval=scale*tf.cast(tf.sqrt(3.0),dtype),dtype=dtype))
    return param
