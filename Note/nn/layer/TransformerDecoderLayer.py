import tensorflow as tf
from Note.nn.layer.multihead_attention import multihead_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.dropout import dropout
from Note.nn.activation import activation_dict


class TransformerDecoderLayer:
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout_rate: float = 0.1,
                 activation = tf.nn.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 bias: bool = True, dtype='float32'):
        self.self_attn = multihead_attention(nhead, input_size=d_model, use_bias=bias, dtype=dtype)
        self.multihead_attn = multihead_attention(nhead, input_size=d_model, use_bias=bias, dtype=dtype)
        # Implementation of Feedforward model
        self.linear1 = dense(dim_feedforward, d_model, use_bias=bias, dtype=dtype)
        self.dropout = dropout(dropout_rate)
        self.linear2 = dense(d_model, dim_feedforward, use_bias=bias, dtype=dtype)

        self.norm_first = norm_first
        self.norm1 = layer_norm(d_model, epsilon=layer_norm_eps, dtype=dtype)
        self.norm2 = layer_norm(d_model, epsilon=layer_norm_eps, dtype=dtype)
        self.norm3 = layer_norm(d_model, epsilon=layer_norm_eps, dtype=dtype)
        self.dropout1 = dropout(dropout_rate)
        self.dropout2 = dropout(dropout_rate)
        self.dropout3 = dropout(dropout_rate)

        if isinstance(activation, str):
            self.activation = activation_dict[activation]
        else:
            self.activation = activation


    def __call__(
        self,
        tgt,
        memory,
        tgt_mask = None,
        memory_mask = None,
        train_flag=True
    ):

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x,
                  attn_mask=None, train_flag=True):
        x = self.self_attn(x,
                           mask=attn_mask,
                           )[0]
        if train_flag:
            return self.dropout1(x)
        else:
            return x


    # multihead attention block
    def _mha_block(self, x, mem,
                   attn_mask=None, train_flag=True):
        x = self.multihead_attn(x, mem,
                                )[0]
        if train_flag:
            return self.dropout2(x)
        else:
            return x


    # feed forward block
    def _ff_block(self, x, train_flag):
        if train_flag:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout3(x)
        else:
            return self.linear2(self.activation(self.linear1(x)))