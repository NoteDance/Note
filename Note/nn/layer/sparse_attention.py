import tensorflow as tf
import Note.nn.initializer as i


class sparse_attention:
    def __init__(self, weight_shape, weight_initializer='Xavier', dtype='float32', mask_mode=None, mask_params=None):
        self.qw = i.initializer(weight_shape, weight_initializer, dtype)
        self.kw = i.initializer(weight_shape, weight_initializer, dtype)
        self.vw = i.initializer(weight_shape, weight_initializer, dtype)
        self.param = [self.qw, self.kw, self.vw]
        # A string that specifies which sparse mode or mask to use, such as "local_window", "block", "routing", etc.
        self.mask_mode = mask_mode
        # A dictionary that stores the parameters for the sparse mode or mask, such as window_size, block_size, top_k, etc.
        self.mask_params = mask_params
    
    
    def output(self, data1, a, data2=None):
        if data2 is None:
            # Use the same data to compute query, key and value
            query = tf.matmul(data1, self.qw)
            key = tf.matmul(data1, self.kw)
            value = tf.matmul(data1, self.vw)
        else:
            # Use different data to compute query and key/value
            query = tf.matmul(data1, self.qw)
            key = tf.matmul(data2, self.kw)
            value = tf.matmul(data2, self.vw)
        # Reshape and transpose the query, key and value tensors to match the attention head dimension
        query = tf.reshape(query, shape=[query.shape[0], query.shape[1], a, query.shape[2] // a])
        key = tf.reshape(key, shape=[key.shape[0], key.shape[1], a, key.shape[2] // a])
        value = tf.reshape(value, shape=[value.shape[0], value.shape[1], a, value.shape[2] // a])
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        # Compute the scores by dot product of query and key and normalize by the square root of the input dimension
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(data1.shape[2] / a)
        
        # If there is a sparse mode or mask specified, call the corresponding function to get a sparse mask tensor
        if self.mask_mode is not None:
            if self.mask_mode == "local_window":
                sparse_mask = self.local_window_mask(data1.shape[1], scores.dtype.name,
                                                     self.mask_params["window_size"])
            elif self.mask_mode == "block":
                sparse_mask = self.block_mask(data1.shape[1], scores.dtype.name,self.mask_params["block_size"])
            elif self.mask_mode == "routing":
                sparse_mask = self.routing_mask(query,key,scores.dtype.name,self.mask_params["top_k"])
            else:
                raise ValueError("Invalid mask mode: {}".format(self.mask_mode))

            # Apply the sparse mask to the scores by subtracting a large negative value from the masked positions
            scores -= 1e9 * (1 - sparse_mask)

        # Apply softmax to the scores to get the attention weights with shape (batch_size,num_heads,seq_len,seq_len)
        attention_weights = tf.nn.softmax(scores,axis=-1)

        # Use the attention weights to weighted sum the value vectors to get the output vectors with shape (batch_size,num_heads,seq_len,head_size)
        output = tf.matmul(attention_weights,value)

        # Concatenate the output vectors to form a tensor with shape (batch_size,seq_len,num_heads * head_size)
        output = tf.transpose(output,perm=[0,2,1,3])
        output = tf.reshape(output,shape=[output.shape[0],output.shape[1],-1])

        return output,attention_weights

    # A function that generates a local window mask where each position only attends to the previous and next window_size positions
    def local_window_mask(self,seq_len,dtype,window_size):
        # Create a local window mask with shape (seq_len,
        # seq_len) that only keeps the window_size positions around the diagonal
        window_mask = tf.linalg.band_part(tf.ones((seq_len,seq_len)),window_size,window_size)
        return window_mask

    # A function that generates a block mask where the sequence is divided into several blocks and each position only attends to the positions within the same block
    def block_mask(self,seq_len,dtype,block_size):
        # Create a block mask with shape (seq_len,
        # seq_len) that only keeps the positions within the same block
        block_mask = tf.linalg.band_part(tf.ones((seq_len // block_size,seq_len // block_size)),0,0)
        block_mask = tf.repeat(tf.repeat(block_mask,block_size,axis=0),block_size,axis=1)
        return block_mask
    
    # A function that generates a routing mask where each position only attends to the top_k most relevant positions
    def routing_mask(self, query, key, dtype, top_k):
        # Compute the dot product of query and key with shape (batch_size, num_heads, seq_len_q, seq_len_k)
        qk = tf.matmul(query, key, transpose_b=True)
        # For each position, find the top_k largest values and their indices with shape (batch_size, num_heads, seq_len_q, top_k)
        values, indices = tf.math.top_k(qk, k=top_k)
        # Create a sparse mask tensor with shape (batch_size * num_heads * seq_len_q, seq_len_k) that only keeps the top_k positions
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len_q = query.shape[2]
        seq_len_k = key.shape[2]
        # Compute the row indices for each position in the sparse tensor
        row_indices = tf.range(batch_size * num_heads * seq_len_q)
        row_indices = tf.repeat(row_indices, top_k)
        # Compute the column indices for each position in the sparse tensor
        col_indices = tf.reshape(indices, shape=[-1])
        # Create a sparse tensor with values of 1 and shape (batch_size * num_heads * seq_len_q, seq_len_k)
        sparse_mask = tf.sparse.SparseTensor(indices=tf.cast(tf.stack([row_indices, col_indices], axis=1), dtype=tf.int64), values=tf.ones_like(col_indices, dtype=dtype), dense_shape=[batch_size * num_heads * seq_len_q, seq_len_k])
        # Reshape the sparse tensor to (batch_size, num_heads, seq_len_q, seq_len_k)
        sparse_mask = tf.sparse.reshape(sparse_mask, shape=[batch_size, num_heads, seq_len_q, seq_len_k])
        sparse_mask = tf.sparse.reorder(sparse_mask)
        return tf.sparse.to_dense(sparse_mask)
