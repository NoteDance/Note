import tensorflow as tf


class ConvRNN:
    def __init__(self,conv_layer,rnn_layer):
        # Receive a convolution layer object and an RNN layer object as parameters
        self.conv_layer=conv_layer
        self.rnn_layer=rnn_layer
        self.output_size=rnn_layer.output_size
    
    
    def __call__(self,data):
        # Get the number of timesteps in the input data
        timestep=data.shape[1]
        # Create an empty list to store the convolution results for each timestep
        conv_outputs=[]
        # Perform convolution operations on the input data for each timestep and add the results to the list
        for i in range(timestep):
            conv_output=self.conv_layer(data[:,i])
            conv_outputs.append(conv_output)
        # Convert the list to a tensor with shape [batch_size, timestep, ...]
        conv_outputs=tf.stack(conv_outputs,axis=1)
        # Pass the convolution results to the RNN layer and get the final output
        rnn_output=self.rnn_layer(conv_outputs)
        return rnn_output