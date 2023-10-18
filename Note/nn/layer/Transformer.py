import tensorflow as tf
from Note.nn.layer.TransformerEncoder import TransformerEncoder
from Note.nn.layer.TransformerDecoder import TransformerDecoder
from Note.nn.layer.TransformerEncoderLayer import TransformerEncoderLayer
from Note.nn.layer.TransformerDecoderLayer import TransformerDecoderLayer
from Note.nn.layer.layer_normalization import layer_normalization


class Transformer:
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = tf.nn.relu,
                 custom_encoder = None, custom_decoder = None,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 bias: bool = True, dtype='float32'):
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, norm_first,
                                                    bias)
            encoder_norm = layer_normalization(d_model, epsilon=layer_norm_eps, dtype=dtype)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, norm_first,
                                                    bias)
            decoder_norm = layer_normalization(d_model, epsilon=layer_norm_eps, dtype=dtype)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead


    def output(self, src, tgt, src_mask = None, tgt_mask = None, memory_mask = None):
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output
