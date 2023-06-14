import tensorflow as tf


class DropPath:
    def __init__(self,rate=0.5,seed=None):
        self.rate=rate
        self.seed=seed
    

    def output(self,data,training=None):
        if training:
            # get the batch size and the number of samples
            batch_size=tf.shape(data)[0]
            num_samples=tf.reduce_prod(tf.shape(data)[1:])
            # create a random mask of shape (batch_size, num_samples)
            mask=tf.random.uniform([batch_size,num_samples],seed=self.seed)>self.rate
            # reshape the mask to match the input shape
            mask=tf.reshape(mask,tf.shape(data))
            # scale the inputs by the inverse of the keep probability
            data=data/(1-self.rate)
            # apply the mask to the inputs
            output=data*tf.cast(mask,data.dtype)
        return output