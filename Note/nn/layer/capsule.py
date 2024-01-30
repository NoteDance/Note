import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.initializer import initializer_

epsilon = 1e-9

class capsule:
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs, vec_len, input_shape=None, kernel_size=None, stride=None, with_routing=True, layer_type='FC', iter_routing=3, steddev=0.01):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.with_routing = with_routing
        self.layer_type = layer_type
        self.iter_routing = iter_routing
        self.steddev = steddev
        if input_shape is not None:
            if self.layer_type == 'CONV':
                if not self.with_routing:
                    self.capsules = conv2d(self.num_outputs * self.vec_len, input_size=input_shape[-1],
                                           kernel_size=self.kernel_size, strides=self.stride, padding="VALID",
                                           activation='relu')
            if self.layer_type == 'FC':
                if self.with_routing:
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    self.b_IJ = tf.zeros([input_shape[0], input_shape[1], self.num_outputs, 1, 1])
                    self.shape = (input_shape[0], -1, 1, input_shape[-2], 1)
                    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
                    self.W = initializer_([1, self.shape[1], vec_len * num_outputs] + self.shape[-2:], 
                                          ['normal', 0.0, steddev], 
                                          tf.float32)
                    self.biases = initializer_((1, 1, num_outputs, vec_len, 1), 'Xavier', tf.float32)
    
    
    def build(self):
        if self.layer_type == 'CONV':
            if not self.with_routing:
                self.capsules = conv2d(self.num_outputs * self.vec_len, input_size=self.input_shape[-1],
                                       kernel_size=self.kernel_size, strides=self.stride, padding="VALID",
                                       activation='relu')
        if self.layer_type == 'FC':
            if self.with_routing:
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                self.b_IJ = tf.zeros([self.input_shape[0], self.input_shape[1], self.num_outputs, 1, 1])
                self.shape = (self.input_shape[0], -1, 1, self.input_shape[-2], 1)
                # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
                self.W = initializer_([1, self.shape[1], self.vec_len * self.num_outputs] + self.shape[-2:], 
                                      ['normal', 0.0, self.steddev], 
                                      tf.float32)
                self.biases = initializer_((1, 1, self.num_outputs, self.vec_len, 1), 'Xavier', tf.float32)
        return
    

    def output(self, data):
        if self.input_shape is None:
            self.input_shape = data.shape
            self.build()
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        if self.layer_type == 'CONV':
            if not self.with_routing:
                capsules = self.capsules.output(data)
                capsules = tf.reshape(capsules, (data.shape[0], -1, self.vec_len, 1))
                capsules = self.squash(capsules)
                return (capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                data = tf.reshape(data, shape=(data.shape[0], -1, 1, data.shape[-2], 1))

                capsules = self.routing(data, self.b_IJ, num_outputs=self.num_outputs, num_dims=self.vec_len)
                capsules = tf.squeeze(capsules, axis=1)

            return (capsules)


    def routing(self, input, b_IJ, num_outputs=10, num_dims=16):
        ''' The routing algorithm.
    
        Args:
            input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
                   shape, num_caps_l meaning the number of capsule in the layer l.
            num_outputs: the number of output capsules.
            num_dims: the number of dimensions for output capsule.
        Returns:
            A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
            representing the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer l, and
            v_j the vector output of capsule j in the layer l+1.
         '''
    
        # Eq.2, calc u_hat
        # Since tf.matmul is a time-consuming op,
        # A better solution is using element-wise multiply, reduce_sum and reshape
        # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
        # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
        # reshape to [a, c]
        input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
        # assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]
    
        u_hat = tf.reduce_sum(self.W * input, axis=3, keepdims=True)
        u_hat = tf.reshape(u_hat, shape=[-1, self.shape[1], num_outputs, num_dims, 1])
        # assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]
    
        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
    
        # line 3,for r iterations do
        for r_iter in range(self.iter_routing):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == self.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + self.biases
                # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]

                # line 6:
                # squash using Eq.1,
                v_J = self.squash(s_J)
                # assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < self.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + self.biases
                v_J = self.squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, self.shape[1], 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v
    
        return(v_J)
    
    
    def squash(self, vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return(vec_squashed)