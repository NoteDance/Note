import tensorflow as tf


class stochastic_depth:
    def __init__(self, drop_path_rate):
        self.drop_path_rate=drop_path_rate
        self.train_flag=True
    
    
    def __call__(self, x, train_flag=True):
        self.train_flag=train_flag
        if train_flag:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x