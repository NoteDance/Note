import tensorflow as tf
import Note.nn.initializer as i


class embedding:
    def __init__(self,embed_size,input_size=None,initializer='Xavier',dtype='float32'):
        self.embed_size=embed_size
        self.input_size=input_size
        self.initializer=initializer
        self.dtype=dtype
        self.output_size=embed_size
        if input_size!=None:
            self.embeddings=i.initializer([input_size,embed_size],initializer,dtype)
            self.param=[self.embeddings]
    
    
    def build(self):
        self.embeddings=i.initializer([self.input_size,self.embed_size],self.initializer,self.dtype)
        self.param=[self.embeddings]
        return
    
    
    def output(self,data):
        output=tf.nn.embedding_lookup(self.embeddings,data)
        return output
