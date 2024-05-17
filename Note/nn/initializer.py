import tensorflow as tf
from Note.nn.Module import Module


def initializer(shape,initializer,dtype='float32',name=None):
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
        elif initializer[0]=='Orthogonal':
            if len(initializer)==3:
                orthogonal=Orthogonal(initializer[1],initializer[2])
            elif len(initializer)==2:
                orthogonal=Orthogonal(initializer[1])
            else:
                orthogonal=Orthogonal()
            param=tf.Variable(orthogonal(shape,dtype))
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
        elif initializer=='Orthogonal':
            orthogonal=Orthogonal()
            param=tf.Variable(orthogonal(shape,dtype))
    if name!=None:
        param.name=name
    return param


def initializer_(shape,initializer,dtype='float32',name=None):
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
        elif initializer[0]=='Orthogonal':
            if len(initializer)==3:
                orthogonal=Orthogonal(initializer[1],initializer[2])
            elif len(initializer)==2:
                orthogonal=Orthogonal(initializer[1])
            else:
                orthogonal=Orthogonal()
            param=tf.Variable(orthogonal(shape,dtype))
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
        elif initializer=='Orthogonal':
            orthogonal=Orthogonal()
            param=tf.Variable(orthogonal(shape,dtype))
    if name!=None:
        param.name=name
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


class Orthogonal:
    """Initializer that generates an orthogonal matrix.

    Also available via the shortcut function `tf.keras.initializers.orthogonal`.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Args:
      gain: multiplicative factor to apply to the orthogonal matrix
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.

    References:
      - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
    """

    def __init__(self, gain=1.0, seed=7):
        self.gain = gain
        self.seed = seed
        self._random_generator = tf.random.Generator.from_seed(seed)

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized to an orthogonal matrix.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`)
        """
        # Check the shape
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be "
                "at least two-dimensional. Received: "
                f"shape={shape} of rank {len(shape)}."
            )
        return self._generate_init_val(shape, dtype)

    def _generate_init_val(self, shape, dtype):
        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

        # Generate a random matrix
        a = self._random_generator.normal(flat_shape, dtype=dtype)
        # Compute the qr factorization
        q, r = tf.linalg.qr(a, full_matrices=False)
        # Make Q uniform
        d = tf.linalg.tensor_diag_part(r)
        q *= tf.sign(d)
        if num_rows < num_cols:
            q = tf.linalg.matrix_transpose(q)
        return self.gain * tf.reshape(q, shape)
