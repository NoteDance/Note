import tensorflow as tf
from Note.nn.initializer import initializer


# Define a custom layer that implements FAVOR+ attention
class FAVOR_attention:
  def __init__(self, dim, nb_heads, nb_random_features, weight_initializer='Xavier', dtype='float32'):
    self.dim = dim # Dimension of the input and output vectors
    self.nb_heads = nb_heads # Number of attention heads
    # Initialize the random features matrix
    self.random_features = initializer([nb_heads, nb_random_features // nb_heads, dim // nb_heads], weight_initializer, dtype)
    # Initialize the query, key and value matrices
    self.qw = initializer([dim , dim] , weight_initializer , dtype)
    self.kw = initializer([dim , dim] , weight_initializer , dtype)
    self.vw = initializer([dim , dim] , weight_initializer , dtype)
    self.output_size = dim * nb_heads
    self.param=[self.qw, self.kw, self.vw, self.random_features]
    

  def output(self, data1, data2=None):
    # Compute the softmax kernel approximation using FAVOR+
    data1 = data1 / tf.math.sqrt(tf.cast(self.dim, data1.dtype.name)) # Scale the data by the square root of the dimension
    # Compute the query, key and value vectors
    query = tf.matmul(data1 , self.qw) # Multiply the data by the query matrix
    if data2 is not None:
      # If data2 is given, compute the key and value vectors from data2
      data2 = data2 / tf.math.sqrt(tf.cast(self.dim, data2.dtype.name)) # Scale the data by the square root of the dimension
      key = tf.matmul(data2 , self.kw) # Multiply the data by the key matrix
      value = tf.matmul(data2 , self.vw) # Multiply the data by the value matrix
    else:
      # If data2 is not given, compute the key and value vectors from data1
      key = tf.matmul(data1 , self.kw) # Multiply the data by the key matrix
      value = tf.matmul(data1 , self.vw) # Multiply the data by the value matrix
    # Split the input into multiple heads
    query = tf.reshape(query, shape=(-1, query.shape[1], self.nb_heads, self.dim // self.nb_heads)) # Reshape the query to have multiple heads
    query = tf.transpose(query, perm=(0, 2, 1, 3)) # Transpose the query to match the attention head order
    key = tf.reshape(key, shape=(-1, key.shape[1], self.nb_heads, self.dim // self.nb_heads)) # Reshape the key to have multiple heads
    key = tf.transpose(key, perm=(0, 2, 1, 3)) # Transpose the key to match the attention head order
    value = tf.reshape(value, shape=(-1, value.shape[1], self.nb_heads, self.dim // self.nb_heads)) # Reshape the value to have multiple heads
    value = tf.transpose(value, perm=(0, 2, 1, 3)) # Transpose the value to match the attention head order
    # Compute omega and kernel using matrix multiplication
    omega = tf.nn.relu(tf.matmul(key, self.random_features, transpose_b=True)) # Multiply the key by the random features matrix and apply a ReLU activation
    omega = omega / tf.reduce_sum(omega, axis=-1, keepdims=True) # Normalize omega by its sum along the last axis
    query=tf.transpose(query, [0, 1, 3, 2]) # Transpose the query to match the omega order
    kernel = tf.matmul(query , omega) # Multiply the query by omega to get the kernel approximation
    kernel=tf.transpose(kernel, perm=[0,1,3,2]) # Transpose the kernel back to its original order
    value = tf.transpose(value, perm=[0,1,3,2]) # Transpose the value to match the kernel order
    # Compute the attention output
    output = tf.matmul(kernel, value) # Multiply the kernel by the value to get the attention output
    # Merge the output back to one head
    output = tf.reshape(output, shape=(-1, data1.shape[1], self.dim * self.nb_heads)) # Reshape the output to merge the attention heads
    return output
