from Note.nn.layer.FAVOR_attention import FAVOR_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization


# Define a custom layer that implements a performer block
class Performer:
  def __init__(self, output_size, nb_heads, nb_random_features, input_size=None, weight_initializer='Xavier', bias_initializer='zeros', activation='relu', use_bias=True, dtype='float32'):
    self.nb_heads=nb_heads
    self.nb_random_features=nb_random_features
    self.input_size=input_size
    self.weight_initializer=weight_initializer
    self.bias_initializer=bias_initializer
    self.activation=activation
    self.use_bias=use_bias
    self.dtype=dtype
    self.train_flag=True
    self.output_size=output_size
    if input_size!=None:
        # Initialize the sublayers
        self.attention = FAVOR_attention(output_size, nb_heads, nb_random_features, input_size, weight_initializer)
        self.ffn_attn = dense([output_size * nb_heads, output_size], weight_initializer, bias_initializer, None, dtype, use_bias)
        self.ffn1 = dense([output_size, output_size * 4], weight_initializer, bias_initializer, activation, dtype, use_bias)
        self.ffn2 = dense([output_size * 4, output_size], weight_initializer, bias_initializer,None, dtype, use_bias)
        self.param=[self.attention.param, self.ffn_attn.param, self.ffn1.param, self.ffn2.param]
 
    
  def build(self):
      # Initialize the sublayers
      self.attention = FAVOR_attention(self.output_size, self.nb_heads, self.nb_random_features, self.input_size, self.weight_initializer)
      self.ffn_attn = dense([self.output_size * self.nb_heads, self.output_size], self.weight_initializer, self.bias_initializer, None, self.use_bias, self.dtype)
      self.ffn1 = dense([self.output_size, self.output_size * 4], self.weight_initializer, self.bias_initializer, self.activation, self.use_bias, self.dtype)
      self.ffn2 = dense([self.output_size * 4, self.output_size], self.weight_initializer, self.bias_initializer, None, self.use_bias, self.dtype)
      self.param=[self.attention.param, self.ffn_attn.param, self.ffn1.param, self.ffn2.param]
      return
    

  def output(self, data, train_flag=True):
    self.train_flag=train_flag
    # Apply the attention sublayer
    attn = self.attention.output(data)
    ffn_attn = self.ffn_attn.output(attn)
    # Apply the residual connection and layer normalization
    output = layer_normalization(ffn_attn + data, train_flag=self.train_flag)
    # Apply the feed-forward sublayer
    ffn1 = self.ffn1.output(output)
    ffn2 = self.ffn2.output(ffn1)
    # Apply the residual connection and layer normalization
    output = layer_normalization(ffn2+output, train_flag=self.train_flag)
    return output
