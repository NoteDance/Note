import tensorflow as tf
from Note.nn.initializer import initializer_

class vector_quantizer:
  def __init__(
      self,
      embedding_dim: int,
      num_embeddings: int,
      commitment_cost: float,
      dtype = 'float32',
  ):
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    self._embedding_shape = [embedding_dim, num_embeddings]
    self._embedding_dtype = dtype

  @property
  def embeddings(self):
    return initializer_(self._embedding_shape, 
                        ['VarianceScaling',1.0,'fan_in','uniform'], 
                        self._embedding_dtype)

  def output(self, data, is_training):
    flat_inputs = tf.reshape(data, [-1, self.embedding_dim])

    distances = (
        tf.math.reduce_sum(tf.math.square(flat_inputs), 1, keepdims=True) -
        2 * tf.matmul(flat_inputs, self.embeddings) +
        tf.math.reduce_sum(tf.math.square(self.embeddings), 0, keepdims=True))

    encoding_indices = tf.math.argmax(-distances, 1)
    encodings = tf.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distances.dtype)

    encoding_indices = tf.reshape(encoding_indices, data.shape[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = tf.math.reduce_mean(
        tf.math.square(tf.stop_gradient(quantized) - data))
    q_latent_loss = tf.math.reduce_mean(
        tf.math.square(quantized - tf.stop_gradient(data)))
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    quantized = data + tf.stop_gradient(quantized - data)
    avg_probs = tf.math.reduce_mean(encodings, 0)
    perplexity = tf.math.exp(-tf.math.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

    return {
        "quantize": quantized,
        "loss": loss,
        "perplexity": perplexity,
        "encodings": encodings,
        "encoding_indices": encoding_indices,
        "distances": distances,
    }

  def quantize(self, encoding_indices):
    w = tf.transpose(self.embeddings, [1, 0])
    return w[(encoding_indices,)]