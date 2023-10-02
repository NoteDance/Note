import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.Module import Module


class embedding(Module):
    def __init__(self,embed_size,input_size=None,initializer='Xavier',trainable=True,dtype='float32'):
        self.embed_size=embed_size
        self.input_size=input_size
        self.initializer=initializer
        self.trainable=trainable
        self.dtype=dtype
        self.output_size=embed_size
        if input_size!=None:
            self.embeddings=i.initializer([input_size,embed_size],initializer,dtype)
            if trainable==True:
                self.param=[self.embeddings]
            else:
                self.param=[]
            Module.param.extend(self.param)
    
    
    def build(self):
        self.embeddings=i.initializer([self.input_size,self.embed_size],self.initializer,self.dtype)
        if self.trainable==True:
            self.param=[self.embeddings]
        else:
            self.param=[]
        Module.param.extend(self.param)
        return
    
    
    def output(self,data):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        output=tf.nn.embedding_lookup(self.embeddings,data)
        return output
