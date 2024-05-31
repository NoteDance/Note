import tensorflow as tf
from Note import nn
from Note.nn.Model import Model


class embedding:
    def __init__(self,output_size,input_size=None,initializer='normal',sparse=False,use_one_hot_matmul=False,trainable=True,dtype='float32'):
        self.input_size=input_size
        self.initializer=initializer
        self.sparse=sparse
        self.use_one_hot_matmul=use_one_hot_matmul
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=output_size
        if input_size!=None:
            self.embeddings=nn.initializer([input_size,output_size],initializer,dtype,trainable)
            self.param=[self.embeddings]
            Model.param.extend(self.param)
    
    
    def build(self):
        self.embeddings=nn.initializer([self.input_size,self.output_size],self.initializer,self.dtype,self.trainable)
        self.param=[self.embeddings]
        Model.param.extend(self.param)
        return
    
    
    def __call__(self, data):
        if data.dtype != "int32" and data.dtype != "int64":
            data = tf.cast(data, "int32")
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        if isinstance(data, tf.sparse.SparseTensor):
            if self.sparse:
                # get sparse embedding values
                embedding_values = tf.nn.embedding_lookup(
                    params=self.embeddings, ids=data.values
                )
                embedding_values = tf.reshape(embedding_values, [-1])
                # get sparse embedding indices
                indices_values_embed_axis = tf.range(self.output_size)
                repeat_times = [data.indices.shape[0]]
                indices_values_embed_axis = tf.expand_dims(
                    tf.tile(indices_values_embed_axis, repeat_times), -1
                )
                indices_values_embed_axis = tf.cast(
                    indices_values_embed_axis, dtype=tf.int64
                )
                current_indices = tf.repeat(
                    data.indices, [self.output_size], axis=0
                )
                new_indices = tf.concat(
                    [current_indices, indices_values_embed_axis], 1
                )
                new_shape = tf.concat(
                    [tf.cast(data.shape, dtype=tf.int64), [self.output_size]],
                    axis=-1,
                )
                out = tf.SparseTensor(
                    indices=new_indices,
                    values=embedding_values,
                    dense_shape=new_shape,
                )
            else:
                sparse_inputs_expanded = tf.sparse.expand_dims(data, axis=-1)
                out = tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=self.embeddings,
                    sparse_ids=sparse_inputs_expanded,
                    default_id=0,
                )
        elif self.use_one_hot_matmul:
            # Note that we change the dtype of the one_hot to be same as the
            # weight tensor, since the input data are usually ints, and weights
            # are floats. The nn.embedding_lookup support ids as ints, but
            # the one_hot matmul need both inputs and weights to be same dtype.
            one_hot_data = tf.one_hot(
                data, depth=self.input_size, dtype=self.dtype
            )
            out = tf.matmul(one_hot_data, self.embeddings)
        else:
            out = tf.gather(self.embeddings, data)

        if self.sparse and not isinstance(out, tf.SparseTensor):
            out = tf.sparse.from_dense(out)
        return out