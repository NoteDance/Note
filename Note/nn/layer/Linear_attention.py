import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization


# Define a kernel function, here we use Gaussian kernel
def gaussian_kernel(x, y, sigma=1.0):
  # x and y are tensors of shape (batch_size, seq_len, dim)
  # sigma is a scalar tensor
  # return a tensor of shape (batch_size, seq_len, seq_len)
  x_norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) # (batch_size, seq_len, 1)
  y_norm = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True) # (batch_size, 1, seq_len)
  xy = tf.matmul(x, y, transpose_b=True) # (batch_size, seq_len, seq_len)
  dist = x_norm + y_norm - 2 * xy # (batch_size, seq_len, seq_len)
  return tf.exp(-dist / (2 * sigma ** 2)) # (batch_size, seq_len, seq_len)


# Define a linear attention layer with multi-head support and kernel approximation
class Linear_attention:
  def __init__(self, output_size, num_heads, kernel_function='gaussian', kernel_approximation='low_rank'):
    if kernel_function=='gaussian': # the kernel function to use
        self.kernel_function=gaussian_kernel
    else:
        self.kernel_function = kernel_function
    self.num_heads = num_heads # the number of attention heads to use
    self.head_dim = output_size // num_heads # the dimension of each head
    self.kernel_approximation = kernel_approximation # the kernel approximation method to use
    self.param=[]
    self.q_dense_list=[]
    self.k_dense_list=[]
    self.v_dense_list=[]
    self.output_size = output_size
    # Use a list to store the projection layers for each head
    for i in range(num_heads):
        self.q_dense_list.append(dense((self.head_dim, self.head_dim), activation=None)) # the query projection layers
        self.k_dense_list.append(dense((self.head_dim, self.head_dim), activation=None)) # the key projection layers
        self.v_dense_list.append(dense((self.head_dim, self.head_dim), activation=None)) # the value projection layers
        self.param.append(self.q_dense_list[i].param)
        self.param.append(self.k_dense_list[i].param)
        self.param.append(self.v_dense_list[i].param)
    self.o_dense = dense((output_size, output_size), activation=None) # the output projection layer
    # Initialize some extra parameters or layers according to the kernel approximation method
    self.param.append(self.o_dense.param)
    if kernel_approximation == 'low_rank':
      # Use a low-rank approximation of Gaussian kernel
      # k_tilde = exp(-0.5 * (q - k)^2 / sigma^2)
      #        ~= exp(-0.5 * q^2 / sigma^2) * exp(q * k / sigma^2) * exp(-0.5 * k^2 / sigma^2)
      #        ~= exp(q * B * B^T * k / sigma^2)
      # where B is a low-rank matrix of shape (head_dim, rank)
      self.rank = 32 # the rank of the low-rank matrix
      for i in range(num_heads):
          self.B_list = [tf.Variable(tf.random.normal((self.head_dim, self.rank)), trainable=True) for _ in range(num_heads)] # the low-rank matrices for each head
          self.param.append(self.B_list[i])
      self.sigma = tf.Variable(tf.ones(1), trainable=True) # the scale parameter for Gaussian kernel
      self.param.append(self.sigma)

  
  def output(self, data1, data2=None):
    # x is a tensor of shape (batch_size, seq_len, dim)
    # return a tensor of shape (batch_size, seq_len, dim)
    # Split the input tensor into num_heads sub-tensors along the last dimension
    data1_split = tf.split(data1, self.num_heads, axis=-1) # a list of tensors of shape (batch_size, seq_len, head_dim)
    # Use a loop to compute the attention output for each head
    z_list = [] # a list to store the attention outputs
    for i in range(self.num_heads):
      # Use the output method of your dense class instead of calling the layer directly
      q = self.q_dense_list[i].output(data1_split[i]) # (batch_size, seq_len, head_dim)
      if data2 is not None:
        # If data2 is given, split it and use it to compute key and value
        data2_split = tf.split(data2, self.num_heads, axis=-1) # a list of tensors of shape (batch_size, seq_len, head_dim)
        k = self.k_dense_list[i].output(data2_split[i]) # (batch_size, seq_len, head_dim)
        v = self.v_dense_list[i].output(data2_split[i]) # (batch_size, seq_len, head_dim)
      else:
        # If data2 is not given, use data to compute key and value
        k = self.k_dense_list[i].output(data1_split[i]) # (batch_size, seq_len, head_dim)
        v = self.v_dense_list[i].output(data1_split[i]) # (batch_size, seq_len, head_dim)
      # Use the kernel approximation method to compute k_tilde
      if self.kernel_approximation == 'low_rank':
        # Use a low-rank approximation of Gaussian kernel
        qB = tf.matmul(q, self.B_list[i]) # (batch_size, seq_len, rank)
        kB = tf.matmul(k, self.B_list[i]) # (batch_size, seq_len, rank)
        qB=layer_normalization(qB) # normalize the qB tensor along the last dimension
        kB=layer_normalization(kB) # normalize the kB tensor along the last dimension
        k_tilde = tf.exp(tf.matmul(qB, kB, transpose_b=True) / self.sigma) # (batch_size, seq_len, seq_len)
      else:
        # Use the exact kernel function
        k_tilde = self.kernel_function(q, k) # (batch_size, seq_len, seq_len)
      v_tilde = tf.matmul(k_tilde, v) # (batch_size, seq_len, head_dim)
      k_tilde_sum = tf.reduce_sum(k_tilde, axis=-1, keepdims=True) # (batch_size, seq_len, 1)
      z = v_tilde / k_tilde_sum # (batch_size, seq_len, head_dim)
      z_list.append(z) # append to the list
    # Concatenate the attention outputs along the last dimension
    z_concat = tf.concat(z_list, axis=-1) # (batch_size, seq_len, dim)
    # Use the output method of your dense class instead of calling the layer directly
    z_final = self.o_dense.output(z_concat) # (batch_size, seq_len, dim)
    return z_final
