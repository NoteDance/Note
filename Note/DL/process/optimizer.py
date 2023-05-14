import tensorflow as tf
from tensorflow.python.ops import state_ops
    

class Gradient:
    def __init__(self,lr):
        self.lr=lr
    
    
    def opt(self,gradient,parameter):
        for i in range(len(gradient)):
            state_ops.assign(parameter[i],parameter[i]-self.lr*gradient[i])
        return parameter
    

class Momentum:
    def __init__(self,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.v=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        if self.flag==0:
            self.v=[0 for x in range(len(gradient))]
        for i in range(len(gradient)):
            self.v[i]=self.gamma*self.v[i]+self.lr*gradient[i]
            state_ops.assign(parameter[i],parameter[i]-self.v[i])
        return parameter
    
    
class AdaGrad:
    def __init__(self,lr,epsilon=1e-06):
        self.lr=lr
        self.epsilon=epsilon
        self.s=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        if self.flag==0:
            self.s=[0 for x in range(len(gradient))]
            self.flag=1
        for i in range(len(gradient)):
            self.s[i]=self.s[i]+gradient[i]**2
            state_ops.assign(parameter[i],parameter[i]-self.lr*gradient[i]/tf.sqrt(self.s[i]+self.epsilon))
        return parameter
    

class RMSProp:
    def __init__(self,lr,gamma=0.9,epsilon=1e-06):
        self.lr=lr
        self.gamma=gamma
        self.epsilon=epsilon
        self.s=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        if self.flag==0:
            self.s=[0 for x in range(len(gradient))]
            self.flag=1
        for i in range(len(gradient)):
            self.s[i]=self.gamma*self.s[i]+(1-self.gamma)*gradient[i]**2
            state_ops.assign(parameter[i],parameter[i]-self.lr*gradient[i]/tf.sqrt(self.s[i]+self.epsilon))
        return parameter


class AdaDelta:
    def __init__(self,lr,rho=0.95,epsilon=1e-05):
        self.lr=lr
        self.rho=rho
        self.epsilon=epsilon
        self.s=[]
        self.x=[]
        self.g=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        if self.flag==0:
           self.s=[0 for x in range(len(gradient))]
           self.x=[0 for x in range(len(gradient))]
           self.g=[x for x in range(len(gradient))]
           self.flag=1
        for i in range(len(gradient)):
            self.s[i]=self.rho*self.s[i]+(1-self.rho)*gradient[i]**2
            self.g[i]=tf.sqrt((self.x[i]+self.epsilon)/(self.s[i]+self.epsilon))*gradient[i]
            state_ops.assign(parameter[i],parameter[i]-self.g[i])
            self.x[i]=self.rho*self.x[i]+(1-self.rho)*self.g[i]**2
        return parameter