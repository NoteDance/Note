import tensorflow as tf
import Note.nn.initializer as i


class attention:
    def __init__(self,weight_shape,weight_initializer='Xavier',dtype='float32'):
        self.qw=i.initializer(weight_shape,weight_initializer,dtype)
        self.kw=i.initializer(weight_shape,weight_initializer,dtype)
        self.sw=i.initializer([weight_shape[1],1],weight_initializer,dtype)
        self.param=[self.qw,self.kw,self.sw]
    
    
    def output(self,en_h,de_h,score_en_h=None):
        if score_en_h is None:
            score_en_h=tf.matmul(en_h,self.qw)
        score=tf.matmul(tf.nn.tanh(score_en_h+tf.expand_dims(tf.matmul(de_h,self.kw),axis=1)),self.sw)
        attention_weights=tf.nn.softmax(score,axis=1)
        if len(en_h.shape)==2:
            context_vector=tf.squeeze(tf.matmul(tf.transpose(en_h,[1,0]),attention_weights),axis=-1)
        else:
            context_vector=tf.squeeze(tf.matmul(tf.transpose(en_h,[0,2,1]),attention_weights),axis=-1)
        attention_weights=tf.squeeze(attention_weights,axis=-1)
        return context_vector,score_en_h,attention_weights