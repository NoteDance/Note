import tensorflow as tf
from Note.nn.layer.dense import dense


# Define a custom layer that implements local and global attention
class Longformer:
  def __init__(self, dim, num_heads, window_size, global_tokens):
    self.dim = dim # the dimension of the input and output vectors
    self.num_heads = num_heads # the number of attention heads to use
    self.head_dim = dim // num_heads # the dimension of each head
    self.window_size = window_size # the size of the local attention window
    self.global_tokens = global_tokens # the indices of the tokens that use global attention
    self.param=[]
    self.q_dense_list=[]
    self.k_dense_list=[]
    self.v_dense_list=[]
    # Use a list to store the projection layers for each head
    for i in range(num_heads):
        self.q_dense_list.append(dense((self.head_dim, self.head_dim), activation=None)) # the query projection layers
        self.k_dense_list.append(dense((self.head_dim, self.head_dim), activation=None)) # the key projection layers
        self.v_dense_list.append(dense((self.head_dim, self.head_dim), activation=None)) # the value projection layers
        self.param.append(self.q_dense_list[i].param)
        self.param.append(self.k_dense_list[i].param)
        self.param.append(self.v_dense_list[i].param)
    self.o_dense = dense((dim, dim), activation=None) # the output projection layer
    self.param.append(self.o_dense.param)
    

  def output(self, data):
    # data is a tensor of shape (batch_size, seq_len, dim)
    # return a tensor of shape (batch_size, seq_len, dim)
    # Split the input tensor into num_heads sub-tensors along the last dimension
    data_split = tf.split(data, self.num_heads, axis=-1) # a list of tensors of shape (batch_size, seq_len, head_dim)
    # Use a loop to compute the attention output for each head
    z_list = [] # a list to store the attention outputs
    for i in range(self.num_heads):
      # Use the output method of your dense class instead of calling the layer directly
      q = self.q_dense_list[i].output(data_split[i]) # (batch_size, seq_len, head_dim)
      k = self.k_dense_list[i].output(data_split[i]) # (batch_size, seq_len, head_dim)
      v = self.v_dense_list[i].output(data_split[i]) # (batch_size, seq_len, head_dim)

      # Compute the local attention mask by sliding a window along the sequence dimension
      mask = tf.linalg.band_part(tf.ones((data.shape[1], data.shape[1])), -self.window_size // 2, self.window_size // 2) # (seq_len, seq_len)
      mask = tf.expand_dims(mask, axis=0) # (1, seq_len, seq_len)
      mask = tf.tile(mask, multiples=(data.shape[0], 1, 1)) # (batch_size, seq_len, seq_len)

      # Compute the global attention mask by setting the global tokens to 1 and others to 0
      global_mask = tf.one_hot(self.global_tokens, depth=data.shape[1]) # (num_global_tokens, seq_len)
      global_mask = tf.reduce_sum(global_mask, axis=0) # (seq_len,)
      global_mask = tf.expand_dims(global_mask, axis=0) # (1, seq_len)
      global_mask = tf.expand_dims(global_mask, axis=-1) # (1, seq_len, 1)
      global_mask = tf.tile(global_mask, multiples=(data.shape[0], 1, data.shape[1])) # (batch_size, seq_len, seq_len)

      # Combine the local and global attention masks
      mask = tf.math.maximum(mask, global_mask) # (batch_size, seq_len, seq_len)

      # Compute the attention scores by dot product of query and key
      scores = tf.matmul(q, k, transpose_b=True) # (batch_size, seq_len, seq_len)

      # Apply the attention mask to the scores
      scores = scores * mask # (batch_size, seq_len, seq_len)

      # Apply softmax along the last dimension
      scores = tf.nn.softmax(scores, axis=-1) # (batch_size, seq_len, seq_len)

      # Compute the attention output
      z = tf.matmul(scores, v) # (batch_size, seq_len, head_dim)

      # Append the output to the list
      z_list.append(z)
    # Concatenate the outputs along the last dimension
    z = tf.concat(z_list, axis=-1) # (batch_size, seq_len, dim)
    # Use the output method of your dense class instead of calling the layer directly
    z = self.o_dense.output(z) # (batch_size, seq_len, dim)
    return z