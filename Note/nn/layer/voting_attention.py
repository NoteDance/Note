import tensorflow as tf
from Note.nn.layer.dense import dense


class voting_attention:
  """Voting Attention layer.

  Args:
    num_heads: The number of attention heads.
    head_size: Per-head hidden size.
    weight_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
  """

  def __init__(self,
               n_head,
               head_size,
               input_size=None,
               weight_initializer="Xavier",
               bias_initializer="zeros",
               use_bias=True,
               dtype='float32'
               ):
    self._num_heads = n_head
    self._head_size = head_size
    self.input_size = input_size
    self._weight_initializer = weight_initializer
    self._bias_initializer = bias_initializer
    self.use_bias=use_bias
    self.dtype=dtype
    if input_size!=None:
        self._query_dense=dense(n_head*head_size,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
        self._key_dense=dense(n_head*head_size,input_size,weight_initializer=weight_initializer,bias_initializer=bias_initializer,use_bias=use_bias,dtype=dtype)
        self.param=[self._query_dense.param,self._key_dense.param]
        
  def build(self):
        self._query_dense=dense(self._num_heads*self._head_size,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self._key_dense=dense(self._num_heads*self._head_size,self.input_size,weight_initializer=self.weight_initializer,bias_initializer=self.bias_initializer,use_bias=self.use_bias,dtype=self.dtype)
        self.param=[self._query_dense.param,self._key_dense.param]

  def output(self, encoder_outputs, doc_attention_mask):
    if encoder_outputs.dtype!=self.dtype:
        encoder_outputs=tf.cast(encoder_outputs,self.dtype)
    if self.input_size==None:
        self.input_size=encoder_outputs.shape[-1]
        self.build()
        
    num_docs = get_shape_list(encoder_outputs, expected_rank=[4])[1]
    cls_embeddings = encoder_outputs[:, :, 0, :]
    key = self._key_dense.output(cls_embeddings)
    n_batch, n_ctx, n_state = key.shape
    key = tf.reshape(key, [n_batch, n_ctx, self._num_heads, -1])
    query = self._query_dense.output(cls_embeddings)
    n_batch, n_ctx, n_state = query.shape
    query = tf.reshape(query, [n_batch, n_ctx, self._num_heads, -1])
    doc_attention_mask = tf.cast(doc_attention_mask, self.dtype)

    key = tf.einsum("BANH,BA->BANH", key, doc_attention_mask)
    query = tf.einsum("BANH,BA->BANH", query, doc_attention_mask)
    attention_matrix = tf.einsum("BXNH,BYNH->BNXY", query, key)
    mask = tf.ones([num_docs, num_docs], self.dtype)
    mask = tf.linalg.set_diag(mask, tf.zeros(num_docs, self.dtype))
    attention_matrix = tf.einsum("BNXY,XY->BNXY", attention_matrix, mask)
    doc_attention_probs = tf.einsum("BNAY->BNA", attention_matrix)
    doc_attention_probs = tf.einsum("BNA->BA", doc_attention_probs)
    infadder = (1.0 - doc_attention_mask) * -100000.0
    return tf.nn.softmax(doc_attention_probs + infadder)

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape