import tensorflow as tf
import Note.nn.initializer as i


class attention:
    def __init__(self,weight_shape,weight_initializer='uniform',dtype='float64'):
        self.qw=i.initializer(weight_shape,weight_initializer,dtype)
        self.kw=i.initializer(weight_shape,weight_initializer,dtype)
        self.sw=i.initializer([weight_shape[0],1],weight_initializer,dtype)
        self.weight_list=[self.attention_w1,self.attention_w2,self.attention_w3]
    
    
    def output(self,en_h,de_h,score_en_h=None):
        if score_en_h is None:
            score_en_h=tf.einsum('ijk,kl->ijl',en_h,self.qw)
        score=tf.einsum('ijk,kl->ijl',tf.nn.tanh(score_en_h+tf.expand_dims(tf.matmul(de_h,self.kw),axis=1)),self.sw)
        attention_weights=tf.nn.softmax(score,axis=1)
        context_vector=tf.reduce_sum(attention_weights*en_h,axis=1)
        return context_vector,score_en_h,attention_weights