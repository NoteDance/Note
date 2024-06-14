import tensorflow as tf
from Note import nn


def add_ml_decoder_head(model):
    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # most CNN models, like Resnet50
        model.global_pool = nn.identity()
        del model.fc
        num_classes = model.num_classes
        num_features = model.num_features
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features)
    elif hasattr(model, 'global_pool') and hasattr(model, 'classifier'):  # EfficientNet
        model.global_pool = nn.identity()
        del model.classifier
        num_classes = model.num_classes
        num_features = model.num_features
        model.classifier = MLDecoder(num_classes=num_classes, initial_num_features=num_features)
    elif 'RegNet' in model._get_name() or 'TResNet' in model._get_name():  # hasattr(model, 'head')
        del model.head
        num_classes = model.num_classes
        num_features = model.num_features
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features)
    else:
        print("Model code-writing is not aligned currently with ml-decoder")
        exit(-1)
    if hasattr(model, 'drop_rate'):  # Ml-Decoder has inner dropout
        model.drop_rate = 0
    return model


class TransformerDecoderLayerOptimal:
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation=tf.nn.relu,
                 layer_norm_eps=1e-5) -> None:
        self.norm1 = nn.layer_norm(d_model, epsilon=layer_norm_eps)
        self.dropout = nn.dropout(dropout)
        self.dropout1 = nn.dropout(dropout)
        self.dropout2 = nn.dropout(dropout)
        self.dropout3 = nn.dropout(dropout)

        self.multihead_attn = nn.multihead_attention(nhead, d_model, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.dense(dim_feedforward, d_model)
        self.linear2 = nn.dense(d_model, dim_feedforward)

        self.norm2 = nn.layer_norm(d_model, epsilon=layer_norm_eps)
        self.norm3 = nn.layer_norm(d_model, epsilon=layer_norm_eps)

        self.activation = activation

    def __call__(self, tgt, memory, tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None):
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# class ExtrapClasses(object):
#     def __init__(self, num_queries: int, group_size: int):
#         self.num_queries = num_queries
#         self.group_size = group_size

#     def __call__(self, h, class_embed_w, class_embed_b, out_extrap):
#         h = tf.tile(tf.expand_dims(h, axis=-1), (1, 1, 1, self.group_size))
#         h = tf.expand_dims(h, axis=-1)
#         h = tf.repeat(h, repeats=self.group_size, axis=-1)
#         w = tf.reshape(class_embed_w, (self.num_queries, h.shape[2], self.group_size))
#         out = tf.reduce_sum((h * w), axis=2) + class_embed_b
#         out = tf.reshape(out, (h.shape[0], self.group_size * self.num_queries))
#         return out

class MLDecoder:
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768, initial_num_features=2048):
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes
        self.embed_len_decoder = embed_len_decoder

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        self.embed_standart = nn.dense(decoder_embedding, initial_num_features)

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)

        # non-learnable queries
        self.query_embed = nn.embedding(decoder_embedding, embed_len_decoder, trainable=False)

        # group fully-connected
        self.num_classes = num_classes
        self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
        self.duplicate_pooling = nn.initializer_(
            (embed_len_decoder, decoder_embedding, self.duplicate_factor), 'Xavier_normal')
        self.duplicate_pooling_bias = nn.initializer_((num_classes), 'zeros')

    def __call__(self, x):
        if len(x.shape) == 4:  # [bs, 7, 7, 2048]
            B, H, W, C = x.shape
            embedding_spatial = tf.reshape(x, (B, H*W, -1))
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.embed_standart(embedding_spatial)
        embedding_spatial_786 = tf.nn.relu(embedding_spatial_786)

        bs = embedding_spatial_786.shape[0]
        query_embed = self.query_embed.embeddings
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = tf.tile(tf.expand_dims(query_embed, axis=1), (1, bs, 1))  # no allocation of memory with expand
        h = self.decoder(tgt, tf.transpose(embedding_spatial_786, (1, 0, 2)))  # [embed_len_decoder, batch, 768]
        h = tf.transpose(h, (1, 0, 2))

        out_extrap = tf.Variable(tf.zeros((h.shape[0], h.shape[1], self.duplicate_factor), dtype=h.dtype))
        for i in range(self.embed_len_decoder):  # group FC
            h_i = h[:, i, :]
            w_i = self.duplicate_pooling[i, :, :]
            out_extrap[:, i, :].assign(tf.matmul(h_i, w_i))
        h_out = tf.reshape(out_extrap, (out_extrap.shape[0], -1))[:, :self.num_classes]
        h_out += self.duplicate_pooling_bias
        logits = h_out
        return logits