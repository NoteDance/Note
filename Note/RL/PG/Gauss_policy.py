import tensorflow as tf
import numpy as np


def Gauss_policy(output,mean,variance):
    return 1/tf.sqrt(2*np.pi*variance)*tf.exp(-((output-mean**2)/2*variance**2))