import tensorflow as tf


class StochasticDepth:
    def __init__(self, pL, layer_depth, total_depth):
        self.pL = pL
        self.layer_depth = layer_depth
        self.survival_prob = 1 - pL * (layer_depth / total_depth) # total_depth is a global variable
    
    
    def output(self, x):
        r = tf.random.uniform([], 0, 1) < self.survival_prob # generate a Bernoulli random number
        if r: # keep the layer
            return x / self.survival_prob
        else: # drop the layer and bypass it with identity function
            return tf.identity(x) * 0
