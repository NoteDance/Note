from Note.nn.layer.Linear_attention import Linear_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization


# Define a linear transformer layer
class Linformer:
  def __init__(self, dim, num_heads, kernel_function='gaussian', kernel_approximation='low_rank'):
    self.attention_layer = Linear_attention(dim, num_heads, kernel_function, kernel_approximation) # the linear attention layer with multi-head support and kernel approximation
    self.ffn_layer = dense((dim, dim), activation=None) # the feed-forward layer
    self.train_flag=True
    self.output_size=dim
    self.param=[self.attention_layer.param, self.ffn_layer.param]
    
  
  def output(self, data, train_flag=True):
    # data is a tensor of shape (batch_size, seq_len, dim)
    # return a tensor of shape (batch_size, seq_len, dim)
    self.train_flag=train_flag
    z_out = self.attention_layer.output(data) # (batch_size, seq_len, dim)
    z_norm = layer_normalization(data + z_out, train_flag=self.train_flag) # (batch_size, seq_len, dim)
    ffn_out = self.ffn_layer.output(z_norm) # (batch_size, seq_len, dim)
    ffn_norm = layer_normalization(z_norm + ffn_out, train_flag=self.train_flag) # (batch_size, seq_len, dim)
    return ffn_norm
