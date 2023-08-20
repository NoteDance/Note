import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.FusedMBConv import FusedMBConv
from Note.nn.layer.MBConv import MBConv
from Note.nn.layer.dense import dense
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class EfficientNetV2B0:
    def __init__(self,classes=1000):
        self.classes=classes
        self.swish=activation_dict['swish']
        self.loss_object=tf.keras.losses.CategoricalCrossentropy()
        self.optimizer=Adam()
        self.km=0
    
    
    def build(self,dtype='float32'):
        self.bc=tf.Variable(0,dtype=dtype)
        self.conv2d=conv2d([3,3,3,32],dtype=dtype)
        self.FusedMBConv1=FusedMBConv(32,16,3,1,1,1,dtype=dtype)
        self.FusedMBConv2=FusedMBConv(16,32,3,2,4,2,dtype=dtype)
        self.FusedMBConv3=FusedMBConv(32,48,3,2,4,2,dtype=dtype)
        self.MBConv1=MBConv(48,96,3,2,4,3,dtype=dtype)
        self.MBConv2=MBConv(96,112,3,1,6,5,dtype=dtype)
        self.MBConv3=MBConv(112,192,3,2,6,8,dtype=dtype)
        self.conv1x1=conv2d([1,1,192,1280],dtype=dtype)
        self.dense=dense([1280,self.classes],dtype=dtype)
        self.param=[self.conv2d.param,
                    self.FusedMBConv1.param,
                    self.FusedMBConv2.param,
                    self.FusedMBConv3.param,
                    self.MBConv1.param,
                    self.MBConv2.param,
                    self.MBConv3.param,
                    self.conv1x1.param,
                    self.dense.param
                    ]
        return
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,'GPU')):
                data=self.conv2d.output(data,strides=[1,2,2,1],padding="SAME")
                data=tf.nn.batch_normalization(data,tf.Variable(tf.zeros([32])),tf.Variable(tf.ones([32])),None,None,1e-5)
                data=self.swish(data)
                data=self.FusedMBConv1.output(data)
                data=self.FusedMBConv2.output(data)
                data=self.FusedMBConv3.output(data)
                data=self.MBConv1.output(data)
                data=self.MBConv2.output(data)
                data=self.MBConv3.output(data)
                data=self.conv1x1.output(data,strides=[1,1,1,1],padding="SAME")
                data=tf.reduce_mean(data,[1,2])
                output=tf.nn.softmax(self.dense.output(data))
        else:
            data=self.conv2d.output(data,strides=[1,2,2,1],padding="SAME")
            data=self.swish(data)
            data=self.FusedMBConv1.output(data,self.km)
            data=self.FusedMBConv2.output(data,self.km)
            data=self.FusedMBConv3.output(data,self.km)
            data=self.MBConv1.output(data,self.km)
            data=self.MBConv2.output(data,self.km)
            data=self.MBConv3.output(data,self.km)
            data=self.conv1x1.output(data,strides=[1,1,1,1],padding="SAME")
            data=tf.reduce_mean(data,[1,2])
            output=tf.nn.softmax(self.dense.output(data))
        return output


    def loss(self,output,labels,p):
        with tf.device(assign_device(p,'GPU')):
            loss_value=self.loss_object(labels,output)
        return loss_value
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,'GPU')):
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param