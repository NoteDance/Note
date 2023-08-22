import tensorflow as tf
import Note.nn.initializer as i


class embedding:
    def __init__(self,vocab_size,embed_size,initializer='Xavier',dtype='float32'):
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.embeddings=i.initializer([vocab_size,embed_size],initializer,dtype)
        self.output_size=embed_size
        self.param=[self.embeddings]
    
    
    def output(self,data):
        output=tf.nn.embedding_lookup(self.embeddings,data)
        return output
