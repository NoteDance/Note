import tensorflow as tf


def reversible_residual(data, f, g):
        data1, data2 = tf.split(data, 2, axis=-1) # split the input into two halves
        output1 = data1 + f.output(data2) # compute the first output half
        output2 = data2 + g.output(output1) # compute the second output half
        output = tf.concat([output1, output2], axis=-1) # concatenate the output halves
        return output