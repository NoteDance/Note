import tensorflow as tf
import numpy as np
import Note.nn.initializer as i
from Note.nn.activation import activation_dict


class ESN:
    def __init__(self, weight_shape, connectivity=0.1, leaky=1.0, spectral_radius=0.9, use_norm2=False, use_bias=True, activation='tanh', weight_initializer='uniform', bias_initializer='zero', dtype='float32'):
        self.units = weight_shape[1] # number of hidden units
        self.connectivity = connectivity # connection probability between two hidden units
        self.leaky = leaky # leaking rate of the reservoir
        self.spectral_radius = spectral_radius # desired spectral radius of recurrent weight matrix
        self.use_norm2 = use_norm2 # whether to use the p-norm function (with p=2) as an upper bound of the spectral radius
        self.use_bias = use_bias # whether to use a bias vector
        self.activation = activation_dict[activation] # activation function
        # Initialize the input-to-hidden weight matrix with uniform distribution
        self.kernel = i.initializer(weight_shape, weight_initializer, dtype) # shape: (input_dim, hidden_size)
        # Initialize the hidden-to-hidden weight matrix with sparse and random values
        recurrent_weights = tf.random.uniform((self.units, self.units), minval=-1.0, maxval=1.0) # shape: (hidden_size, hidden_size)
        recurrent_weights[tf.random.uniform((self.units, self.units)) > self.connectivity] = 0
        # Scale the recurrent weights to satisfy the echo state property
        if self.use_norm2:
          # Use the p-norm as an upper bound of the spectral radius
          recurrent_norm2 = tf.norm(recurrent_weights, ord=2)
          recurrent_weights = recurrent_weights * (self.spectral_radius / recurrent_norm2)
        else:
          # Use the largest eigenvalue as the spectral radius
          recurrent_eigenvalues = tf.py_function(np.linalg.eigvals, [recurrent_weights], Tout=tf.complex64)
          recurrent_spectral_radius = tf.math.reduce_max(tf.math.abs(recurrent_eigenvalues))
          recurrent_weights = recurrent_weights * (self.spectral_radius / recurrent_spectral_radius)
        self.recurrent_kernel = tf.Variable(initial_value=recurrent_weights,
                                            trainable=False,
                                            name='recurrent_kernel') # shape: (hidden_size, hidden_size)
        if self.use_bias:
          # Initialize the bias vector with zeros
          self.bias = i.initializer([self.units], bias_initializer, dtype) # shape: (hidden_size,)
        else:
          self.bias = None
        if use_bias:
          self.weight_list = [self.kernel, self.recurrent_kernel, self.bias]
        else:
          self.weight_list = [self.kernel, self.recurrent_kernel]
    
    
    def output(self, data):
        # Initialize the hidden state with zeros
        state = tf.zeros((tf.shape(data)[0], self.units)) # shape: (batch_size ,hidden_size)
        # Create an empty list to store the outputs
        outputs = []
        # Loop over the time steps of the input sequence
        for t in range(tf.shape(data)[1]):
            # Get the input at time t
            input_t = data[:, t, :] # shape: (batch_size ,input_dim)
            # Compute the new hidden state using the ESN equation
            state = (1 - self.leaky) * state + self.leaky * self.activation(
            tf.matmul(input_t, self.kernel) + tf.matmul(state, self.recurrent_kernel) + (self.bias if self.use_bias else 0)) # shape: (batch_size ,hidden_size)
            # Append the hidden state to the outputs list
            outputs.append(state)
        # Stack the outputs along the time dimension
        outputs = tf.stack(outputs, axis=1) # shape: (batch_size ,seq_len ,hidden_size)
        return outputs
