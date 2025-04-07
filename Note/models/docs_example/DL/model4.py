"""
This example demonstrates how to use Note's PCGrad class 
by modifying the train_step function inherited from the Model class.
"""
import tensorflow as tf
from Note import nn
from Note.nn.optimizer.pcgrad import PCGrad
# from Note.nn.optimizer.pcgrad import PPCGrad

class Model(nn.Model):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add(nn.conv2d(32, 3, activation='relu'))
        self.layers.add(nn.max_pool2d())
        self.layers.add(nn.conv2d(64, 3, activation='relu'))
        self.layers.add(nn.max_pool2d())
        self.layers.add(nn.flatten())
        self.layers.add(nn.dense(64, activation='relu'))
        self.layers.add(nn.dense(10))
        self.pcgrad = PCGrad()
        # self.pcgrad = PPCGrad()
    
    def __call__(self, x):
        return self.layers(x)

    @tf.function(jit_compile=True)
    def train_step(self, train_data, labels, loss_object, train_loss, train_accuracy, optimizer):
        with tf.GradientTape() as tape:
            output = self.__call__(train_data)
            losses = loss_object(labels, output)
        gradients = self.pcgrad(tape, losses, self.param)
        optimizer.apply_gradients(zip(gradients, self.param), tape)
        loss = train_loss(losses)
        if train_accuracy!=None:
            acc=train_accuracy(labels, output)
            return loss,acc
        return loss,None