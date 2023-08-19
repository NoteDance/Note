import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.MBConv import MBConv
from Note.nn.layer.dense import dense
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class EfficientNetB0:
    """A class that implements the EfficientNetB0 model for image classification.

    Args:
        classes: integer, the number of classes to predict. Default is 1000.
    """
    def __init__(self,classes=1000):
        self.classes=classes # store the number of classes
        self.swish=activation_dict['swish'] # get the swish activation function from the activation dictionary
        self.loss_object=tf.keras.losses.CategoricalCrossentropy() # create a categorical crossentropy loss object
        self.optimizer=Adam() # create an Adam optimizer object
        self.km=0
    
    
    def build(self,dtype='float32'):
        """A method that builds the model by creating different layers."""
        self.bc=tf.Variable(0,dtype=dtype) # create a variable to store the batch count
        self.conv2d=conv2d([3,3,3,32],dtype=dtype) # create a conv2d layer with 32 filters and no bias
        self.MBConv1=MBConv(32,16,3,1,1,1,dtype) # create a MBConv layer with 16 output channels and 1 repeat
        self.MBConv2=MBConv(16,24,3,2,6,2,dtype) # create a MBConv layer with 24 output channels and 2 repeats
        self.MBConv3=MBConv(24,40,5,2,6,2,dtype) # create a MBConv layer with 40 output channels and 2 repeats
        self.MBConv4=MBConv(40,80,3,2,6,3,dtype) # create a MBConv layer with 80 output channels and 3 repeats
        self.MBConv5=MBConv(80,112,5,1,6,3,dtype) # create a MBConv layer with 112 output channels and 3 repeats
        self.MBConv6=MBConv(112,192,5,2,6,4,dtype) # create a MBConv layer with 192 output channels and 4 repeats
        self.MBConv7=MBConv(192,320,3,1,6,1,dtype) # create a MBConv layer with 320 output channels and 1 repeat
        self.dense=dense([320,self.classes],dtype=dtype) # create a dense layer with self.classes units
        self.param=[self.conv2d.param,
                    self.MBConv1.param,
                    self.MBConv2.param,
                    self.MBConv3.param,
                    self.MBConv4.param,
                    self.MBConv5.param,
                    self.MBConv6.param,
                    self.MBConv7.param,
                    self.dense.param
                    ] # store all the parameters in a list
        return
    
    
    def fp(self,data,p=None):
        """A method that performs forward propagation on the input data and returns the output tensor.

        Args:
            data: tensor, the input data.
            p: integer, the index of the device to use.

        Returns:
            output: tensor, the output data after applying the model.
        """
        if self.km==1: # if kernel mode is 1
            with tf.device(assign_device(p,'GPU')): # assign the device to use
                data=self.conv2d.output(data,strides=[1,2,2,1],padding="SAME") # apply the conv2d layer with strides 2 and same padding
                data=tf.nn.batch_normalization(data,tf.Variable(tf.zeros([32])),tf.Variable(tf.ones([32])),None,None,1e-5) # apply batch normalization to normalize the output
                data=self.swish(data) # apply swish activation function to increase nonlinearity
                data=self.MBConv1.output(data) # apply the MBConv1 layer
                data=self.MBConv2.output(data) # apply the MBConv2 layer
                data=self.MBConv3.output(data) # apply the MBConv3 layer
                data=self.MBConv4.output(data) # apply the MBConv4 layer
                data=self.MBConv5.output(data) # apply the MBConv5 layer
                data=self.MBConv6.output(data) # apply the MBConv6 layer
                data=self.MBConv7.output(data) # apply the MBConv7 layer
                data=tf.reduce_mean(data,[1,2]) # apply global average pooling to get the mean value of each channel
                output=tf.nn.softmax(self.dense.output(data)) # apply the dense layer and softmax activation function to get the probability distribution of each class
        else:
            data=self.conv2d.output(data,strides=[1,2,2,1],padding="SAME") # apply the conv2d layer with strides 2 and same padding
            data=self.swish(data) # apply swish activation function to increase nonlinearity
            data=self.MBConv1.output(data,self.km) # apply the MBConv1 layer
            data=self.MBConv2.output(data,self.km) # apply the MBConv2 layer
            data=self.MBConv3.output(data,self.km) # apply the MBConv3 layer
            data=self.MBConv4.output(data,self.km) # apply the MBConv4 layer
            data=self.MBConv5.output(data,self.km) # apply the MBConv5 layer
            data=self.MBConv6.output(data,self.km) # apply the MBConv6 layer
            data=self.MBConv7.output(data,self.km) # apply the MBConv7 layer
            data=tf.reduce_mean(data,[1,2]) # apply global average pooling to get the mean value of each channel
            output=tf.nn.softmax(self.dense.output(data)) # apply the dense layer and softmax activation function to get the probability distribution of each class 
        return output # return the output tensor
    
    
    def loss(self,output,labels,p):
        """A method that calculates the loss value between the output tensor and the labels tensor.

        Args:
            output: tensor, the output data after applying the model.
            labels: tensor, the true labels of the input data.
            p: integer, the index of the device to use.

        Returns:
            loss_value: tensor, the loss value.
        """
        with tf.device(assign_device(p,'GPU')): # assign the device to use
            loss_value=self.loss_object(labels,output) # calculate the loss value using categorical crossentropy loss function
        return loss_value # return the loss value
    
    
    def opt(self,gradient,p):
        """A method that updates the model parameters using the optimizer and the gradient.

        Args:
            gradient: tensor, the gradient of the loss function with respect to the model parameters.
            p: integer, the index of the device to use.

        Returns:
            param: list, the updated model parameters.
        """
        with tf.device(assign_device(p,'GPU')): # assign the device to use
            param=self.optimizer.opt(gradient,self.param,self.bc[0]) # update the model parameters using Adam optimizer and batch count
            return param # return the updated model parameters
