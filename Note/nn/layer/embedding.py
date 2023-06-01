import tensorflow as tf


class EmbeddingLayer:
    def __init__(self,vocab_size,embed_size,initializer=None,dtype=tf.float32):
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        if initializer is None:
            initializer=tf.random.uniform_initializer(-1.0,1.0)
        self.embeddings=tf.Variable(initializer([self.vocab_size,self.embed_size]),dtype=dtype)
    
    
    def __call__(self,inputs):
        outputs=tf.nn.embedding_lookup(self.embeddings,inputs)
        return outputs