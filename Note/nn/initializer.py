import tensorflow as tf
from Note.nn.Module import Module


def initializer(shape,initializer,dtype):
    if type(initializer)==list:
        if initializer[0]=='normal':
            param=tf.Variable(tf.random.normal(shape=shape,mean=initializer[1],stddev=initializer[2],dtype=dtype))
        elif initializer[0]=='uniform':
            param=tf.Variable(tf.random.uniform(shape=shape,minval=initializer[1],maxval=initializer[2],dtype=dtype))
        elif initializer[0]=='VarianceScaling':
            scale=initializer[1]
            fan_in,fan_out=compute_fans(shape)
            if initializer[2]=='fan_in':
                scale/=max(1.0,fan_in)
            elif initializer[2]=='fan_out':
                scale/=max(1.0,fan_out)
            elif initializer[2]=='fan_avg':
                scale/=max(1.0,(fan_in+fan_out)/2.0)
            else:
                raise ValueError('Invalid mode:'+initializer[2])
            if initializer[3]=='truncated_normal':
                stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
                param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
            elif initializer[3]=='uniform':
                limit=tf.cast(tf.sqrt(3.0*scale),dtype)
                param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
            else:
                raise ValueError('Invalid distribution:'+initializer[3])
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
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,(fan_in+fan_out)/2.0)
            limit=tf.cast(tf.sqrt(3.0*scale),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
        elif initializer=='Xavier_normal':
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,(fan_in+fan_out)/2.0)
            stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
            param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
        elif initializer=='He':
            scale=2.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
            param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
        elif initializer=='He_uniform':
            scale=2.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            limit=tf.cast(tf.sqrt(3.0*scale),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
        elif initializer=='Lecun':
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
            param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
        elif initializer=='Lecun_uniform':
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            limit=tf.cast(tf.sqrt(3.0*scale),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
    return param


def initializer_(shape,initializer,dtype):
    if type(initializer)==list:
        if initializer[0]=='normal':
            param=tf.Variable(tf.random.normal(shape=shape,mean=initializer[1],stddev=initializer[2],dtype=dtype))
        elif initializer[0]=='uniform':
            param=tf.Variable(tf.random.uniform(shape=shape,minval=initializer[1],maxval=initializer[2],dtype=dtype))
        elif initializer[0]=='VarianceScaling':
            scale=initializer[1]
            fan_in,fan_out=compute_fans(shape)
            if initializer[2]=='fan_in':
                scale/=max(1.0,fan_in)
            elif initializer[2]=='fan_out':
                scale/=max(1.0,fan_out)
            elif initializer[2]=='fan_avg':
                scale/=max(1.0,(fan_in+fan_out)/2.0)
            else:
                raise ValueError('Invalid mode:'+initializer[2])
            if initializer[3]=='truncated_normal':
                stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
                param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
            elif initializer[3]=='uniform':
                limit=tf.cast(tf.sqrt(3.0*scale),dtype)
                param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
            else:
                raise ValueError('Invalid distribution:'+initializer[3])
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
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,(fan_in+fan_out)/2.0)
            limit=tf.cast(tf.sqrt(3.0*scale),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
        elif initializer=='Xavier_normal':
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,(fan_in+fan_out)/2.0)
            stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
            param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
        elif initializer=='He':
            scale=2.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
            param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
        elif initializer=='He_uniform':
            scale=2.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            limit=tf.cast(tf.sqrt(3.0*scale),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
        elif initializer=='Lecun':
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            stddev=tf.cast(tf.sqrt(scale)/0.87962566103423978,dtype)
            param=tf.Variable(tf.random.truncated_normal(shape=shape,mean=0.0,stddev=stddev,dtype=dtype))
        elif initializer=='Lecun_uniform':
            scale=1.0
            fan_in,fan_out=compute_fans(shape)
            scale/=max(1.0,fan_in)
            limit=tf.cast(tf.sqrt(3.0*scale),dtype)
            param=tf.Variable(tf.random.uniform(shape=shape,minval=-limit,maxval=limit,dtype=dtype))
    Module.param.append(param)
    return param


def compute_fans(shape):
    if len(shape)<1:
        fan_in=fan_out=1
    elif len(shape)==1:
        fan_in=fan_out=shape[0]
    elif len(shape)==2:
        fan_in=shape[0]
        fan_out=shape[1]
    else:
        receptive_field_size=1
        for dim in shape[:-2]:
            receptive_field_size*=dim
        fan_in=shape[-2]*receptive_field_size
        fan_out=shape[-1]*receptive_field_size
    return int(fan_in),int(fan_out)
