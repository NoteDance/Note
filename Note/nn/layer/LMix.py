import tensorflow as tf
import numpy as np


def lmix(x,y,alpha):
    batch_size=x.shape[0]
    lam=np.random.beta(alpha,alpha)
    index=tf.random.shuffle(tf.range(batch_size))
    mask=tf.cast(tf.random.uniform(x.shape)<lam,x.dtype)
    x_mix=x*mask+x[index]*(1-mask)
    kernel=tf.constant([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],dtype=x.dtype)
    kernel=tf.reshape(kernel,[3, 3, 1, 1])
    x_mix=tf.nn.conv2d(x_mix,kernel, strides=[1,1,1,1],padding='SAME')
    y_mix=lam*y+(1-lam)*y[index]
    return x_mix,y_mix