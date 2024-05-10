import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.dropout import dropout


class GCNLayer:
    def __init__(self, in_features, out_features, bias=True):
        self.linear = dense(out_features, in_features, use_bias=bias)

    def __call__(self, x, adj):
        x = self.linear(x)
        return tf.matmul(adj, x)


class GCN:
    def __init__(self, x_dim, h_dim, out_dim, nb_layers=2, dropout_rate=0.5, bias=True):
        layer_sizes = [x_dim] + [h_dim] * nb_layers + [out_dim]
        self.gcn_layers = [
            GCNLayer(in_dim, out_dim, bias)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.dropout = dropout(dropout_rate)

    def __call__(self, x, adj):
        for layer in self.gcn_layers[:-1]:
            x = tf.nn.relu(layer(x, adj))
            x = self.dropout(x)

        x = self.gcn_layers[-1](x, adj)
        return x