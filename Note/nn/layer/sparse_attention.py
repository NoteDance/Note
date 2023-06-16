import tensorflow as tf
import Note.nn.initializer as i


class sparse_attention:
    def __init__(self, weight_shape, weight_initializer='Xavier', dtype='float32'):
        self.qw = i.initializer(weight_shape, weight_initializer, dtype)
        self.kw = i.initializer(weight_shape, weight_initializer, dtype)
        self.vw = i.initializer(weight_shape, weight_initializer, dtype)
        self.param = [self.qw, self.kw, self.vw]
    
    
    def output(self, data, a, mask=None):
        query = tf.sparse.sparse_dense_matmul(tf.sparse.reshape(data, shape=[-1,data.shape[2]]), self.qw)
        key = tf.sparse.sparse_dense_matmul(tf.sparse.reshape(data, shape=[-1,data.shape[2]]), self.kw)
        value = tf.sparse.sparse_dense_matmul(tf.sparse.reshape(data, shape=[-1,data.shape[2]]), self.vw)
        query = tf.reshape(query, shape=[data.shape[0], data.shape[1], data.shape[2]])
        query = tf.reshape(query, shape=[query.shape[0], query.shape[1], a, data.shape[2] // a])
        key = tf.reshape(key, shape=[data.shape[0], data.shape[1], data.shape[2]])
        key = tf.reshape(key, shape=[key.shape[0], key.shape[1], a, data.shape[2] // a])
        value = tf.reshape(query, shape=[data.shape[0], data.shape[1], data.shape[2]])
        value = tf.reshape(query, shape=[value.shape[0], value.shape[1], a, data.shape[2] // a])
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(data.shape[2] / a)
        if mask is not None:
            scores += mask * -1e9
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, shape=[output.shape[0], output.shape[1], -1])
        return output, attention_weights
