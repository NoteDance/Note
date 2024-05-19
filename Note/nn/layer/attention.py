import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.Model import Model


class attention: # define a class for attention mechanism
    def __init__(self, use_scale=False, score_mode="dot", dtype='float32'):
        self.use_scale = use_scale
        self.score_mode = score_mode
        self.dtype=dtype
        self.param=[]
        if use_scale:
            self.scale = initializer((),'ones',dtype)
            self.param.append(self.scale)
        if score_mode == "concat":
            self.concat_score_weight = initializer((),'ones',dtype)
            self.param.append(self.concat_score_weight)
        Model.param.extend(self.param)
    
    
    def __call__(self, query, value, key=None): # define the output method
        if query.dtype!=self.dtype:
            query=tf.cast(query,self.dtype)
        if value.dtype!=self.dtype:
            value=tf.cast(value,self.dtype)
        if key is not None and key.dtype!=self.dtype:
            key=tf.cast(key,self.dtype)
        if self.score_mode == "dot":
            if key==None:
                scores = tf.matmul(query, value, transpose_b=True)
            else:
                scores = tf.matmul(query, key, transpose_b=True)
            if self.scale is not None:
                scores *= self.scale
        elif self.score_mode == "concat":
            # Reshape tensors to enable broadcasting.
            # Reshape into [batch_size, Tq, 1, dim].
            q_reshaped = tf.expand_dims(query, axis=-2)
            # Reshape into [batch_size, 1, Tv, dim].
            if key==None:
                k_reshaped = tf.expand_dims(value, axis=-3)
            else:
                k_reshaped = tf.expand_dims(key, axis=-3)
            if self.scale is not None:
                scores = self.concat_score_weight * tf.reduce_sum(
                    tf.tanh(self.scale * (q_reshaped + k_reshaped)), axis=-1
                )
            else:
                scores = self.concat_score_weight * tf.reduce_sum(
                    tf.tanh(q_reshaped + k_reshaped), axis=-1
                )
        distribution = tf.nn.softmax(scores)
        return tf.matmul(distribution, value)