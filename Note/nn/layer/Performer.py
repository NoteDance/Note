from Note.nn.layer.FAVOR_attention import FAVOR_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization


# Define a custom layer that implements a performer block
class Performer:
  def __init__(self, dim, nb_heads, nb_random_features, weight_initializer='Xavier', bias_initializer='zeros', activation='relu', dtype='float32', use_bias=True):
    # Initialize the sublayers
    self.attention = FAVOR_attention(dim, nb_heads, nb_random_features)
    self.ffn_attn = dense([dim * nb_heads, dim], weight_initializer, bias_initializer, None, dtype, use_bias)
    self.ffn1 = dense([dim, dim * 4], weight_initializer, bias_initializer, activation, dtype, use_bias)
    self.ffn2 = dense([dim * 4, dim], weight_initializer, bias_initializer,None, dtype, use_bias)
    self.output_size=dim
    self.param=[self.attention.param, self.ffn_attn.param, self.ffn1.param, self.ffn2.param]
    

  def output(self, data, train_flag=True):
    # Apply the attention sublayer
    attn = self.attention.output(data)
    ffn_attn = self.ffn_attn.output(attn)
    # Apply the residual connection and layer normalization
    output = layer_normalization(ffn_attn + data, train_flag=train_flag)
    # Apply the feed-forward sublayer
    ffn1 = self.ffn1.output(output)
    ffn2 = self.ffn2.output(ffn1)
    # Apply the residual connection and layer normalization
    output = layer_normalization(ffn2+output, train_flag=train_flag)
    return output
