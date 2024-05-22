import tensorflow as tf
import math

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = tf.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = tf.linalg.band_part(tf.ones((L, S), dtype=tf.bool), -1, 0)
        attn_bias = tf.where(temp_mask, attn_bias, float("-inf"))
        attn_bias = tf.cast(attn_bias, query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            attn_bias = tf.where(attn_mask, attn_bias, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = tf.matmul(query, tf.transpose(key, (0, 1, 3, 2))) * scale_factor
    attn_weight += attn_bias
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    attn_weight = tf.nn.dropout(attn_weight, dropout_p)
    return tf.matmul(attn_weight, value)