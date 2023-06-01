import tensorflow as tf
from tensorflow.python.ops import state_ops
    

class Gradient:
    def __init__(self,lr):
        self.lr=lr
    
    
    def opt(self,gradient,parameter):
        for i in range(len(gradient)):
            state_ops.assign(parameter[i],parameter[i]-self.lr*gradient[i])
        return
    

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
        return
    
    
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
        return
    

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
        return


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
    
    
    def opt(self,gradient,parameter,t):
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
            state_ops.assign(parameter[i],parameter[i]-self.g[i])
        return


class Nadam:
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
        self.m=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        if self.flag==0:
            self.v=[0 for x in range(len(gradient))]
            self.s=[0 for x in range(len(gradient))]
            self.v_=[x for x in range(len(gradient))]
            self.s_=[x for x in range(len(gradient))]
            self.g=[x for x in range(len(gradient))]
            self.m=[x for x in range(len(gradient))]
            self.flag+=1
        for i in range(len(gradient)):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*gradient[i]
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*gradient[i]**2
            self.v_[i]=self.v[i]/(1-self.beta1**(t+1))
            self.s_[i]=self.s[i]/(1-self.beta2**(t+1))
            self.m[i]=(self.beta1*gradient[i])/(1-self.beta1**(t+1))
            self.g[i]=self.lr*(self.m[i]+(1-self.beta1)*gradient[i])/(tf.sqrt(self.s_[i])+self.epsilon)
            state_ops.assign(parameter[i],parameter[i]-self.g[i])
        return


class AdaMax:
    def __init__(self,learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07):
        self.learning_rate=learning_rate
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon
        self.v=[]
        self.u=[]
        self.g=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        if self.flag==0:
            self.v=[0 for x in range(len(gradient))]
            self.u=[0 for x in range(len(gradient))]
            self.g=[x for x in range(len(gradient))]
            self.flag+=1
        for i in range(len(gradient)):
            self.v[i]=self.beta_1*self.v[i]+(1-self.beta_1)*gradient[i]
            self.u[i]=tf.maximum(self.beta_2*self.u[i],tf.abs(gradient[i]))
            self.g[i]=self.learning_rate/(1-self.beta_1**(t+1))*self.v[i]/(self.u[i]+self.epsilon)
            state_ops.assign(parameter[i],parameter[i]-self.g[i])
        return


class Ftrl:
    def __init__(self,learning_rate=0.001,learning_rate_power=-0.5,initial_accumulator_value=0.1,l1_regularization_strength=0.0,l2_regularization_strength=0.0,l2_shrinkage_regularization_strength=0.0,beta=0.0):
        self.learning_rate=learning_rate
        self.learning_rate_power=learning_rate_power
        self.initial_accumulator_value=initial_accumulator_value
        self.l1_regularization_strength=l1_regularization_strength
        self.l2_regularization_strength=l2_regularization_strength
        self.l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength
        self.beta=beta
        self.n=[]
        self.sigma=[]
        self.z=[]
        self.g=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        if self.flag==0:
            self.n=[self.initial_accumulator_value for x in range(len(gradient))]
            self.sigma=[0 for x in range(len(gradient))]
            self.z=[0 for x in range(len(gradient))]
            self.g=[x for x in range(len(gradient))]
            self.flag+=1
        for i in range(len(gradient)):
            prev_n=self.n[i]
            self.n[i]=self.n[i]+gradient[i]**2
            self.sigma[i]=(self.n[i]**-self.learning_rate_power-prev_n**-self.learning_rate_power)/self.learning_rate
            self.z[i]=self.z[i]+gradient[i]-self.sigma[i]*parameter[i]
            if tf.abs(self.z[i])<self.l1_regularization_strength:
                state_ops.assign(parameter[i],tf.zeros_like(self.z[i]))
            else:
                state_ops.assign(parameter[i],(tf.sign(self.z[i])*self.l1_regularization_strength-self.z[i])/((self.beta+tf.sqrt(self.n[i]))/self.learning_rate+self.l2_regularization_strength))
        return
