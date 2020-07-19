import tensorflow as tf
from tensorflow.python.ops import state_ops
    

class Gradient:
    def __init__(self,lr):
        self.lr=lr
    
    
    def Gradient(self,gradient,variable):
        for i in range(len(gradient)):
            state_ops.assign(variable[i],variable[i]-self.lr*gradient[i])
        return
    

class Momentum:
    def __init__(self,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.v=[]
        self.flag=0
    
    
    def Momentum(self,gradient,variable):
        if self.flag==0:
            self.v=[0 for x in range(len(gradient))]
        for i in range(len(gradient)):
            self.v[i]=self.gamma*self.v[i]+self.lr*gradient[i]
            state_ops.assign(variable[i],variable[i]-self.v[i])
        return
    
    
class AdaGrad:
    def __init__(self,lr,epsilon=1e-06):
        self.lr=lr
        self.epsilon=epsilon
        self.s=[]
        self.flag=0
    
    
    def AdaGrad(self,gradient,variable):
        if self.flag==0:
            self.s=[0 for x in range(len(gradient))]
            self.flag=1
        for i in range(len(gradient)):
            self.s[i]=self.s[i]+gradient[i]**2
            state_ops.assign(variable[i],variable[i]-self.lr*gradient[i]/tf.sqrt(self.s[i]+self.epsilon))
        return
    

class RMSProp:
    def __init__(self,lr,gamma,epsilon=1e-06):
        self.lr=lr
        self.gamma=gamma
        self.epsilon=epsilon
        self.s=[]
        self.flag=0
    
    
    def RMSProp(self,gradient,variable):
        if self.flag==0:
            self.s=[0 for x in range(len(gradient))]
            self.flag=1
        for i in range(len(gradient)):
            self.s[i]=self.gamma*self.s[i]+(1-self.gamma)*gradient[i]**2
            state_ops.assign(variable[i],variable[i]-self.lr*gradient[i]/tf.sqrt(self.s[i]+self.epsilon))
        return


class AdaDelta:
    def __init__(self,lr,rho,epsilon=1e-05):
        self.lr=lr
        self.rho=rho
        self.epsilon=epsilon
        self.s=[]
        self.x=[]
        self.g=[]
        self.flag=0
    
    
    def AdaDelta(self,gradient,variable):
        if self.flag==0:
           self.s=[0 for x in range(len(gradient))]
           self.x=[0 for x in range(len(gradient))]
           self.g=[x for x in range(len(gradient))]
           self.flag=1
        for i in range(len(gradient)):
            self.s[i]=self.rho*self.s[i]+(1-self.rho)*gradient[i]**2
            self.g[i]=tf.sqrt((self.x[i]+self.epsilon)/(self.s[i]+self.epsilon))*gradient[i]
            state_ops.assign(variable[i],variable[i]-self.g[i])
            self.x[i]=self.rho*self.x[i]+(1-self.rho)*self.g[i]**2
        return
    
    
class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-07):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.v=[]
        self.s=[]
        self.v_=[]
        self.s_=[]
        self.g=[]
        self.flag=0
    
    
    def Adam(self,gradient,variable,t):
        if self.flag==0:
            self.v=[0 for x in range(len(gradient))]
            self.s=[0 for x in range(len(gradient))]
            self.v_=[x for x in range(len(gradient))]
            self.s_=[x for x in range(len(gradient))]
            self.g=[x for x in range(len(gradient))]
            self.flag+=1
        for i in range(len(gradient)):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*gradient[i]
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*gradient[i]**2
            self.v_[i]=self.v[i]/(1-self.beta1**(t+1))
            self.s_[i]=self.s[i]/(1-self.beta2**(t+1))
            self.g[i]=self.lr*self.v_[i]/(tf.sqrt(self.s_[i])+self.epsilon)
            state_ops.assign(variable[i],variable[i]-self.g[i])
        return