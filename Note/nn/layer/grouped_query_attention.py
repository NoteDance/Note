import tensorflow as tf
from Note.nn.layer.einsum_dense import einsum_dense
from Note.nn.layer.dropout import dropout
from Note.nn.layer.softmax import softmax


class grouped_query_attention:
    """Grouped Query Attention layer.

    This is an implementation of grouped-query attention introduced by
    [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245). Here
    `num_key_value_heads` denotes number of groups, setting
    `num_key_value_heads` to 1 is equivalent to multi-query attention, and
    when `num_key_value_heads` is equal to `num_query_heads` it is equivalent
    to multi-head attention.

    This layer first projects `query`, `key`, and `value` tensors. Then, `key`
    and `value` are repeated to match the number of heads of `query`.

    Then, the `query` is scaled and dot-producted with `key` tensors. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities and concatenated back to a single
    tensor.

    Args:
        head_dim: Size of each attention head.
        num_query_heads: Number of query attention heads.
        num_key_value_heads: Number of key and value attention heads.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        weight_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.

    Call arguments:
        query: Query tensor of shape `(batch_dim, target_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `target_seq_len` is the length of
            target sequence, and `feature_dim` is dimension of feature.
        value: Value tensor of shape `(batch_dim, source_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `source_seq_len` is the length of
            source sequence, and `feature_dim` is dimension of feature.
        key: Optional key tensor of shape
            `(batch_dim, source_seq_len, feature_dim)`. If not given, will use
            `value` for both `key` and `value`, which is most common case.
        attention_mask: A boolean mask of shape
            `(batch_dim, target_seq_len, source_seq_len)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, where 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output
            should be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Will go with either using the training mode of the parent
            layer/model or `False` (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: Result of the computation, of shape
            `(batch_dim, target_seq_len, feature_dim)`, where `target_seq_len`
            is for target sequence length and `feature_dim` is the query input
            last dim.
        attention_scores: (Optional) attention coefficients of shape
            `(batch_dim, num_query_heads, target_seq_len, source_seq_len)`.
    """

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        query_shape=None,
        value_shape=None,
        key_shape=None,
        dropout_rate=0.0,
        use_bias=True,
        weight_initializer="Xavier",
        bias_initializer="zeros",
    ):
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "`num_query_heads` must be divisible"
                " by `num_key_value_heads`."
            )
        self.num_repeats = num_query_heads // num_key_value_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.query_shape = query_shape
        self.value_shape = value_shape
        self.key_shape = value_shape if key_shape is None else key_shape
        if query_shape!=None:
            self.query_dense=einsum_dense("bqm,muh->bquh", (query_shape[0], self.num_query_heads, self.head_dim), 
                                          input_shape=query_shape, bias_axes="uh" if self.use_bias else None,
                                          weight_initializer=weight_initializer, bias_initializer=bias_initializer)
            self.key_dense=einsum_dense("bkm,mvh->bkvh", (query_shape[0], self.num_key_value_heads, self.head_dim), 
                                          input_shape=key_shape, bias_axes="vh" if self.use_bias else None,
                                          weight_initializer=weight_initializer, bias_initializer=bias_initializer)
            self.value_dense=einsum_dense("bkm,mvh->bkvh", (query_shape[0], self.num_key_value_heads, self.head_dim), 
                                          input_shape=value_shape, bias_axes="vh" if self.use_bias else None,
                                          weight_initializer=weight_initializer, bias_initializer=bias_initializer)
            self.softmax=softmax(axis=-1)
            self.dropout_layer=dropout(dropout_rate)
            self.dot_product_equation = "bquh,bkuh->buqk"
            self.combine_equation = "buqk,bkuh->bquh"
            self.output_dense=einsum_dense("bquh,uhm->bqm", (query_shape[0], self.feature_dim), 
                                          input_shape=(query_shape[0], query_shape[1], self.num_query_heads, self.head_dim), bias_axes="m" if self.use_bias else None,
                                          weight_initializer=weight_initializer, bias_initializer=bias_initializer)
        

    def build(self):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        self.query_dense=einsum_dense("bqm,muh->bquh", (self.query_shape[0], self.num_query_heads, self.head_dim), 
                                      input_shape=self.query_shape, bias_axes="uh" if self.use_bias else None,
                                      weight_initializer=self.weight_initializer, bias_initializer=self.bias_initializer)
        self.key_dense=einsum_dense("bkm,mvh->bkvh", (self.query_shape[0], self.num_key_value_heads, self.head_dim), 
                                      input_shape=self.key_shape, bias_axes="vh" if self.use_bias else None,
                                      weight_initializer=self.weight_initializer, bias_initializer=self.bias_initializer)
        self.value_dense=einsum_dense("bkm,mvh->bkvh", (self.query_shape[0], self.num_key_value_heads, self.head_dim), 
                                      input_shape=self.value_shape, bias_axes="vh" if self.use_bias else None,
                                      weight_initializer=self.weight_initializer, bias_initializer=self.bias_initializer)
        self.softmax=softmax(axis=-1)
        self.dropout_layer=dropout(self.dropout_rate)
        self.dot_product_equation = "bquh,bkuh->buqk"
        self.combine_equation = "buqk,bkuh->bquh"
        self.output_dense=einsum_dense("bquh,uhm->bqm", (self.query_shape[0], self.feature_dim), 
                                      input_shape=(self.query_shape[0], self.query_shape[1], self.num_query_heads, self.head_dim), bias_axes="m" if self.use_bias else None,
                                      weight_initializer=self.weight_initializer, bias_initializer=self.bias_initializer)
        return


    def __call__(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        query=tf.cast(query,'float32')
        value=tf.cast(value,'float32')
        key=tf.cast(key,'float32')
        if self.query_shape==None:
            self.query_shape=query.shape
            self.value_shape=value.shape
            self.key_shape=key.shape
            self.build()
            
        if key is None:
            key = value

        attention_mask = self.compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        key = tf.repeat(
            key, self.num_repeats, axis=2
        )  # (batch_dim, source_seq_len, query_heads, head_dim)
        value = tf.repeat(
            value, self.num_repeats, axis=2
        )  # (batch_dim, source_seq_len, query_heads, head_dim)

        output, scores = self.compute_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            training=training,
        )

        output = self.output_dense(
            output
        )  # (batch_dim, target_seq_len, feature_dim)

        if return_attention_scores:
            return output, scores
        return output

    def compute_attention_mask(
        self,
        query,
        value,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_causal_mask=False,
    ):
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, T, N, key_dim)`.
            value: Projected value tensor of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            use_causal_mask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `use_causal_mask=True`.
        """
        auto_mask = None
        if query_mask is not None:
            query_mask = tf.cast(query_mask, "bool")  # defensive casting
            # B = batch size, T = max query length
            auto_mask = tf.expand_dims(query_mask, -1)  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = tf.cast(value_mask, "bool")  # defensive casting
            # B = batch size, S == max value length
            mask = tf.expand_dims(value_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = tf.cast(key_mask, "bool")  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = tf.expand_dims(key_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self.compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else tf.cast(attention_mask, bool) & auto_mask
            )
        return attention_mask

    def compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean tensor equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query tensor of shape `(B, T, ...)`.
            value: value tensor of shape `(B, S, ...)` (optional, defaults to
                query).

        Returns:
            mask: a boolean tensor of shape `(1, T, S)` containing a lower
                triangular matrix of shape `(T, S)`.
        """
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        ones_mask = tf.ones((1, q_seq_length, v_seq_length), dtype="int32")
        row_index = tf.math.cumsum(ones_mask, axis=-2)
        col_index = tf.math.cumsum(ones_mask, axis=-1)
        return tf.math.greater_equal(row_index, col_index)

    def compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        query = tf.math.multiply(
            query,
            1.0 / tf.math.sqrt(tf.cast(self.head_dim, query.dtype)),
        )
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        scores = tf.einsum(
            self.dot_product_equation, query, key
        )  # (batch_dim, query_heads, target_seq_len, source_seq_len)
        scores = self.masked_softmax(scores, attention_mask=attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        scores_dropout = self.dropout_layer(scores, training=training)
        output = tf.einsum(self.combine_equation, scores_dropout, value)
        return output, scores

    def masked_softmax(self, scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # scores = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -1 * 2 - 1
            for _ in range(len(scores.shape) - len(attention_mask.shape)):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self.softmax(scores, mask=attention_mask)