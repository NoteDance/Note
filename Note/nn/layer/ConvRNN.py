import tensorflow as tf


class ConvRNN:
    def __init__(self,conv_layer,rnn_layer):
        # Receive a convolution layer object and an RNN layer object as parameters
        self.conv_layer=conv_layer
        self.rnn_layer=rnn_layer
        # Merge the weight lists of the two layer objects into one param list
        self.param=conv_layer.weight_list+rnn_layer.weight_list
    
    
    def output(self,data):
        # Get the number of timesteps in the input data
        timestep=data.shape[1]
        # Create an empty list to store the convolution results for each timestep
        conv_outputs=[]
        # Perform convolution operations on the input data for each timestep and add the results to the list
        for i in range(timestep):
            conv_output=self.conv_layer.output(data[:,i])
            conv_outputs.append(conv_output)
        # Convert the list to a tensor with shape [batch_size, timestep, ...]
        conv_outputs=tf.stack(conv_outputs,axis=1)
        # Pass the convolution results to the RNN layer and get the final output
        rnn_output=self.rnn_layer.output(conv_outputs)
        return rnn_output