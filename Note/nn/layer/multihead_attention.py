import tensorflow as tf
import Note.nn.initializer as i


class multihead_attention:
    def __init__(self, weight_shape, num_heads, weight_initializer='Xavier', dtype='float32'):
        # Create weight matrices for query, key, value, and output using i.initializer function
        self.qw = i.initializer(weight_shape, weight_initializer, dtype)
        self.kw = i.initializer(weight_shape, weight_initializer, dtype)
        self.vw = i.initializer(weight_shape, weight_initializer, dtype)
        self.ow = i.initializer([weight_shape[1], weight_shape[1]], weight_initializer, dtype)
        # Add all weight matrices to model parameters
        self.param = [self.qw, self.kw, self.vw, self.ow]
        # Define the number of heads and the dimension of each head
        self.num_heads = num_heads
        self.head_dim = weight_shape[1] // num_heads
    

    def output(self, query, key, value, mask=None):
        # Linearly transform query, key, and value to obtain new query, key, and value
        query = tf.matmul(query, self.qw)  # shape: (batch_size, seq_length_q, dim)
        key = tf.matmul(key, self.kw)  # shape: (batch_size, seq_length_k, dim)
        value = tf.matmul(value, self.vw)  # shape: (batch_size, seq_length_v, dim)
        # Split the new query, key, and value into multiple heads to obtain multihead query, key, and value
        query = tf.reshape(query, shape=[query.shape[0], query.shape[1], self.num_heads, self.head_dim])  # shape: (batch_size, seq_length_q, num_heads, head_dim)
        key = tf.reshape(key, shape=[key.shape[0], key.shape[1], self.num_heads, self.head_dim])  # shape: (batch_size, seq_length_k, num_heads, head_dim)
        value = tf.reshape(value, shape=[value.shape[0], value.shape[1], self.num_heads, self.head_dim])  # shape: (batch_size, seq_length_v, num_heads, head_dim)
        # Transpose the multihead query, key, and value to obtain transposed multihead query, key, and value
        query = tf.transpose(query, perm=[0, 2, 1, 3])  # shape: (batch_size, num_heads, seq_length_q, head_dim)
        key = tf.transpose(key, perm=[0, 2, 1, 3])  # shape: (batch_size, num_heads, seq_length_k, head_dim)
        value = tf.transpose(value, perm=[0, 2, 1, 3])  # shape: (batch_size, num_heads, seq_length_v, head_dim)
        # Apply scaled dot-product attention to the transposed multihead query and key to obtain scaled dot-product attention output and attention weights
        scaled_dot_product_attention_output, scaled_dot_product_attention_weights = self.scaled_dot_product_attention(query=query,
                                                                                                                      key=key,
                                                                                                                      value=value,
                                                                                                                      mask=None)  # shape: (batch_size, num_heads, seq_length_q, head_dim)
        # Transpose the scaled dot-product attention output to obtain transposed scaled dot-product attention output
        scaled_dot_product_attention_output = tf.transpose(scaled_dot_product_attention_output, perm=[0, 2, 1, 3])  # shape: (batch_size, seq_length_q, num_heads, head_dim)
        # Concatenate the transposed scaled dot-product attention output to obtain concatenated scaled dot-product attention output
        concat_scaled_dot_product_attention_output = tf.reshape(scaled_dot_product_attention_output, shape=[scaled_dot_product_attention_output.shape[0], scaled_dot_product_attention_output.shape[1], -1])  # shape: (batch_size, seq_length_q, dim)
        # Linearly transform the concatenated scaled dot-product attention output to obtain the final output
        output = tf.matmul(concat_scaled_dot_product_attention_output, self.ow)  # shape: (batch_size, seq_length_q, weight_shape[1])
        # Return the final output and attention weights
        return output, scaled_dot_product_attention_weights


    def scaled_dot_product_attention(self, query, key, value, mask):
        # Compute the dot product of query and key to obtain the dot product output
        dot_product_output = tf.matmul(query, key, transpose_b=True)  # shape: (batch_size, num_heads, seq_length_q, seq_length_k)
        # Scale the dot product output to obtain the scaled dot product output
        scaled_dot_product_output = dot_product_output / tf.sqrt(tf.cast(self.head_dim, query.dtype.name))  # shape: (batch_size, num_heads, seq_length_q, seq_length_k)
        # If there is a mask, add the mask to the scaled dot product output to obtain the masked scaled dot product output
        if mask is not None:
            scaled_dot_product_output += mask * -1e9  # shape: (batch_size, num_heads, seq_length_q, seq_length_k)
        # Apply softmax to the masked scaled dot product output to obtain the attention weights
        attention_weights = tf.nn.softmax(scaled_dot_product_output, axis=-1)  # shape: (batch_size, num_heads, seq_length_q, seq_length_k)
        # Compute the dot product of attention weights and value to obtain the attention output
        attention_output = tf.matmul(attention_weights, value)  # shape: (batch_size, num_heads, seq_length_q, head_dim)
        # Return the attention output and attention weights
        return attention_output, attention_weights