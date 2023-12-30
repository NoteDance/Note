import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.dropout import dropout

class multi_cls_heads:
  """Pooling heads sharing the same pooling stem."""

  def __init__(self,
               inner_dim,
               cls_list,
               input_size=None,
               cls_token_idx=0,
               activation="tanh",
               dropout_rate=0.0,
               initializer="Xavier",
               dtype='float32'
               ):
    """Initializes the `multi_cls_heads`.

    Args:
      inner_dim: The dimensionality of inner projection layer. If 0 or `None`
        then only the output projection layer is created.
      cls_list: a list of pairs of (the numbers
        of classes.
      cls_token_idx: The index inside the sequence to pool.
      activation: Activation function to use.
      dropout_rate: Dropout probability.
      initializer: Initializer for dense layer kernels.
    """
    self.dropout_rate = dropout_rate
    self.inner_dim = inner_dim
    self.cls_list = cls_list
    self.input_size = input_size
    self.activation = activation
    self.initializer = initializer
    self.cls_token_idx = cls_token_idx
    self.dtype = dtype
    
    if input_size!=None:
        if self.inner_dim:
          self.dense = dense(
              inner_dim,
              input_size,
              activation=self.activation,
              weight_initializer=self.initializer,
              dtype=dtype
              )
        self.dropout = dropout(rate=self.dropout_rate)
        self.out_projs = []
        output_size=self.dense.output_size
        for num_classes in cls_list:
          self.out_projs.append(
              dense(
                  num_classes,
                  output_size,
                  weight_initializer=self.initializer,
                  dtype=dtype
                  ))
          output_size=self.out_projs[-1].output_size
        self.output_size = output_size
  
  def build(self):
    if self.inner_dim:
      self.dense = dense(
          self.inner_dim,
          self.input_size,
          activation=self.activation,
          weight_initializer=self.initializer,
          dtype=self.dtype
          )
    self.dropout = dropout(rate=self.dropout_rate)
    self.out_projs = []
    output_size=self.dense.output_size
    for num_classes in self.cls_list:
      self.out_projs.append(
          dense(
              num_classes,
              output_size,
              weight_initializer=self.initializer,
              dtype=self.dtype
              ))
      output_size=self.out_projs[-1].output_size
    self.output_size = output_size

  def output(self, features: tf.Tensor, only_project: bool = False):
    """Implements call().

    Args:
      features: a rank-3 Tensor when self.inner_dim is specified, otherwise
        it is a rank-2 Tensor.
      only_project: a boolean. If True, we return the intermediate Tensor
        before projecting to class logits.

    Returns:
      If only_project is True, a Tensor with shape= [batch size, hidden size].
      If only_project is False, a dictionary of Tensors.
    """
    if features.dtype!=self.dtype:
        features=tf.cast(features,self.dtype)
    if self.input_size==None:
        self.input_size=features.shape[-1]
        self.build()
    if not self.inner_dim:
      x = features
    else:
      x = features[:, self.cls_token_idx, :]  # take <CLS> token.
      x = self.dense(x)

    if only_project:
      return x
    x = self.dropout(x)

    outputs = {}
    for proj_layer in self.out_projs:
      outputs[proj_layer.name] = proj_layer(x)
    return outputs