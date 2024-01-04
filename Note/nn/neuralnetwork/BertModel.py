import copy
import json
import numpy as np
import six
import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization
from Note.nn.Module import Module


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
  
  seq_length = input_ids.shape[1]
  embedding_size = config.hidden_size

  config = Note.nn.neuralnetwork.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = Note.nn.neuralnetwork.BertModel(config=config, seq_length, is_training=True,
                                input_mask=input_mask, token_type_ids=token_type_ids)
  model.fp(input_ids)

  label_embeddings = Note.nn.initializer.initializer_(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               seq_length,
               is_training,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    
    self.input_mask = input_mask
    self.token_type_ids = token_type_ids

    # Perform embedding lookup on the word ids.
    self.embedding_lookup=embedding_lookup(
        vocab_size=config.vocab_size,
        embedding_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # Add positional embeddings and token type embeddings, then layer
    # normalize and perform dropout.
    self.embedding_postprocessor = embedding_postprocessor(
                                    embedding_size=config.hidden_size,
                                    use_token_type=True,
                                    token_type_ids=token_type_ids,
                                    token_type_vocab_size=config.type_vocab_size,
                                    use_position_embeddings=True,
                                    initializer_range=config.initializer_range,
                                    max_position_embeddings=config.max_position_embeddings,
                                    dropout_prob=config.hidden_dropout_prob)

    self.all_encoder_layers = transformer_model(
        embedding_size=config.hidden_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        intermediate_act_fn=get_activation(config.hidden_act),
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        initializer_range=config.initializer_range,
        do_return_all_layers=True)

    self.pooled_layer = dense(
        config.hidden_size,
        config.hidden_size,
        activation='tanh',
        weight_initializer=['truncated_normal',config.initializer_range])
  
    
  def fp(self,input_ids):
      input_shape = get_shape_list(input_ids, expected_rank=2)
      batch_size = input_shape[0]
      seq_length = input_shape[1]
      if self.input_mask is None:
        self.input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

      if self.token_type_ids is None:
        self.token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
      (self.embedding_output, self.embedding_table) = self.embedding_lookup.output(input_ids)
      self.embedding_output = self.embedding_postprocessor.output(self.embedding_output)
      # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
      # mask of shape [batch_size, seq_length, seq_length] which is used
      # for the attention scores.
      attention_mask = create_attention_mask_from_input_mask(
          input_ids, self.input_mask)
      # Run the stacked transformer.
      # `sequence_output` shape = [batch_size, seq_length, hidden_size].
      self.sequence_output = self.all_encoder_layers.output(self.embedding_output,attention_mask)
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      
      # We "pool" the model by simply taking the hidden state corresponding
      # to the first token. We assume that this has been pre-trained
      self.first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
      self.pooled_output = self.pooled_layer.output(self.first_token_tensor)
      

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.math.tanh(
      (tf.math.sqrt(2 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.math.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


class embedding_lookup:
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  def __init__(self,vocab_size,embedding_size=128,initializer_range=0.02,use_one_hot_embeddings=False):
      self.embedding_table = tf.random.truncated_normal(
          shape=[vocab_size, embedding_size],
          stddev=initializer_range)
      self.vocab_size=vocab_size
      self.embedding_size=embedding_size
      self.use_one_hot_embeddings=use_one_hot_embeddings
      Module.param.extend(self.embedding_table)
      
  def output(self,input_ids):
      # This function assumes that the input is of shape [batch_size, seq_length,
      # num_inputs].
      #
      # If the input is a 2D tensor of shape [batch_size, seq_length], we
      # reshape to [batch_size, seq_length, 1].
      if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])
    
      flat_input_ids = tf.reshape(input_ids, [-1])
      if self.use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
        output = tf.matmul(one_hot_input_ids, self.embedding_table)
      else:
        output = tf.gather(self.embedding_table, flat_input_ids)
    
      input_shape = get_shape_list(input_ids)
    
      output = tf.reshape(output,
                          input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
      return (output, self.embedding_table)


class embedding_postprocessor:                 
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  def __init__(self,
              embedding_size,
              use_token_type=False,
              token_type_ids=None,
              token_type_vocab_size=16,
              use_position_embeddings=True,
              initializer_range=0.02,
              max_position_embeddings=512,
              dropout_prob=0.1):
      self.token_type_table = tf.random.truncated_normal(
              shape=[token_type_vocab_size, embedding_size],
              stddev=initializer_range)
      self.full_position_embeddings = tf.random.truncated_normal(
              shape=[max_position_embeddings, embedding_size],
              stddev=initializer_range)
      self.layer_norm=layer_normalization(embedding_size)
      self.use_token_type=use_token_type
      self.token_type_ids=token_type_ids
      self.token_type_vocab_size=token_type_vocab_size
      self.use_position_embeddings=use_position_embeddings
      self.dropout_prob=dropout_prob
      Module.param.extend([self.token_type_table,self.full_position_embeddings])
      
      
  def output(self,input_tensor):
      input_shape = get_shape_list(input_tensor, expected_rank=3)
      batch_size = input_shape[0]
      seq_length = input_shape[1]
      width = input_shape[2]
    
      output = input_tensor
    
      if self.use_token_type:
        if self.token_type_ids is None:
          raise ValueError("`token_type_ids` must be specified if"
                           "`use_token_type` is True.")
    
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(self.token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings
    
      if self.use_position_embeddings:
          # So `full_position_embeddings` is effectively an embedding table
          # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
          # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
          # perform a slice.
          position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                         [seq_length, -1])
          num_dims = len(output.shape.as_list())
    
          # Only the last two dimensions are relevant (`seq_length` and `width`), so
          # we broadcast among the first dimensions, which is typically just
          # the batch size.
          position_broadcast_shape = []
          for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
          position_broadcast_shape.extend([seq_length, width])
          position_embeddings = tf.reshape(position_embeddings,
                                           position_broadcast_shape)
          output += position_embeddings
    
      output = dropout(self.layer_norm.output(output),self.dropout_prob)
      return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


class attention_layer:
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  def __init__(self,
               width,
               num_attention_heads=1,
               size_per_head=512,
               query_act=None,
               key_act=None,
               value_act=None,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               do_return_2d_tensor=False,
               ):
    self.query_layer=dense(
                    num_attention_heads * size_per_head,
                    width,
                    activation=query_act,
                    weight_initializer=['truncated_normal',initializer_range])
    self.key_layer=dense(
                    num_attention_heads * size_per_head,
                    width,
                    activation=key_act,
                    weight_initializer=['truncated_normal',initializer_range])
    self.value_layer=tf.layers.dense(
                    num_attention_heads * size_per_head,
                    width,
                    activation=value_act,
                    weight_initializer=['truncated_normal',initializer_range])
    self.num_attention_heads=num_attention_heads
    self.size_per_head=size_per_head
    self.attention_probs_dropout_prob=attention_probs_dropout_prob
    self.do_return_2d_tensor=do_return_2d_tensor
    self.output_size=num_attention_heads * size_per_head
  
  def output(self,from_tensor,to_tensor,attention_mask=None):
      def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                               seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])
    
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor
    
      from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
      to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    
      if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")
    
      if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
      elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
          raise ValueError(
              "When passing in rank 2 tensors to attention_layer, the values "
              "for `batch_size`, `from_seq_length`, and `to_seq_length` "
              "must all be specified.")
    
      # Scalar dimensions referenced here:
      #   B = batch size (number of sequences)
      #   F = `from_tensor` sequence length
      #   T = `to_tensor` sequence length
      #   N = `num_attention_heads`
      #   H = `size_per_head`
    
      from_tensor_2d = reshape_to_matrix(from_tensor)
      to_tensor_2d = reshape_to_matrix(to_tensor)
    
      # `query_layer` = [B*F, N*H]
      query_layer = self.query_layer.output(from_tensor_2d)
    
      # `key_layer` = [B*T, N*H]
      key_layer = self.key_layer.output(to_tensor_2d)
    
      # `value_layer` = [B*T, N*H]
      value_layer = self.value_layer.output(to_tensor_2d)
    
      # `query_layer` = [B, N, F, H]
      query_layer = transpose_for_scores(query_layer, batch_size,
                                         self.num_attention_heads, from_seq_length,
                                         self.size_per_head)
    
      # `key_layer` = [B, N, T, H]
      key_layer = transpose_for_scores(key_layer, batch_size, self.num_attention_heads,
                                       to_seq_length, self.size_per_head)
    
      # Take the dot product between "query" and "key" to get the raw
      # attention scores.
      # `attention_scores` = [B, N, F, T]
      attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
      attention_scores = tf.math.multiply(attention_scores,
                                     1.0 / tf.math.sqrt(float(self.size_per_head)))
    
      if self.attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(self.attention_mask, axis=[1])
    
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder
    
      # Normalize the attention scores to probabilities.
      # `attention_probs` = [B, N, F, T]
      attention_probs = tf.nn.softmax(attention_scores)
    
      # This is actually dropping out entire tokens to attend to, which might
      # seem a bit unusual, but is taken from the original Transformer paper.
      attention_probs = dropout(attention_probs, self.attention_probs_dropout_prob)
    
      # `value_layer` = [B, T, N, H]
      value_layer = tf.reshape(
          value_layer,
          [batch_size, to_seq_length, self.num_attention_heads, self.size_per_head])
    
      # `value_layer` = [B, N, T, H]
      value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    
      # `context_layer` = [B, N, F, H]
      context_layer = tf.matmul(attention_probs, value_layer)
    
      # `context_layer` = [B, F, N, H]
      context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    
      if self.do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, self.num_attention_heads * self.size_per_head])
      else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])
    
      return context_layer


class transformer_model:
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  def __init__(self,    
               embedding_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_act_fn=gelu,
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               initializer_range=0.02,
               do_return_all_layers=False):
      
      if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))
    
      attention_head_size = int(hidden_size / num_attention_heads)
      
      attention_output_input_size=0
      self.attention_heads_layers = []
      self.attention_layers = []
      self.intermediate_layers = []
      self.layers = []
      self.layer_norms1 = []
      self.layer_norms2 = []
      for layer_idx in range(num_hidden_layers):
          attention_head = attention_layer(
                embedding_size,
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                do_return_2d_tensor=True)
          self.attention_heads_layers.append(attention_head)
          attention_output_input_size += attention_head.output_size
          attention_output = dense(
                                  hidden_size,
                                  attention_output_input_size,
                                  weight_initializer=['truncated_normal',initializer_range])
          layer_norm = layer_normalization(attention_output.output_size)
          self.attention_layers.append(attention_output)
          self.layer_norms1.append(layer_norm)
          intermediate_output = dense(
                              intermediate_size,
                              attention_output.output_size,
                              activation=intermediate_act_fn,
                              weight_initializer=['truncated_normal',initializer_range])
          self.intermediate_layers.append(intermediate_output)
          layer_output = dense(
                          hidden_size,
                          intermediate_output.output_size,
                          weight_initializer=['truncated_normal',initializer_range])
          layer_norm = layer_normalization(layer_output.output_size)
          self.layers.append(layer_output)
          self.layer_norms2.append(layer_norm)
      self.hidden_size=hidden_size
      self.num_hidden_layers=num_hidden_layers
      self.hidden_dropout_prob=hidden_dropout_prob
      self.do_return_all_layers=do_return_all_layers
          
  def output(self,input_tensor,attention_mask=None):
      input_shape = get_shape_list(input_tensor, expected_rank=3)
      input_width = input_shape[2]
    
      # The Transformer performs sum residuals on all layers so the input needs
      # to be the same as the hidden size.
      if input_width != self.hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, self.hidden_size))
    
      # We keep the representation as a 2D tensor to avoid re-shaping it back and
      # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
      # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
      # help the optimizer.
      prev_output = reshape_to_matrix(input_tensor)
    
      attention_heads = []
      all_layer_outputs = []
      for layer_idx in range(self.num_hidden_layers):
          layer_input = prev_output
            
          attention_head = self.attention_heads_layers[layer_idx].output(layer_input,layer_input,attention_mask)
          attention_heads.append(attention_head)

          if len(attention_heads) == 1:
            attention_output = attention_heads[0]
          else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)

          # Run a linear projection of `hidden_size` then add a residual
          # with `layer_input`.
          attention_output = self.attention_layers[layer_idx].output(attention_output)
          attention_output = dropout(attention_output, self.hidden_dropout_prob)
          attention_output = self.layer_norms1.output(attention_output + layer_input)

          # The activation is only applied to the "intermediate" hidden layer.
          intermediate_output = self.intermediate_layers.output(attention_output)
  
          # Down-project back to `hidden_size` then add the residual.
          layer_output = self.layers.output(intermediate_output)
          layer_output = dropout(layer_output, self.hidden_dropout_prob)
          layer_output = self.layer_norms2.output(layer_output + attention_output)
          prev_output = layer_output
          all_layer_outputs.append(layer_output)
    
      if self.do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
          final_output = reshape_from_matrix(layer_output, input_shape)
          final_outputs.append(final_output)
        return final_outputs
      else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def get_shape_list(tensor, expected_rank=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """

  if expected_rank is not None:
    assert_rank(tensor, expected_rank)

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


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    raise ValueError
