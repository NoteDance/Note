import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest


class Gradient:
    def __init__(self,lr):
        self.lr=lr
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        for i in range(len(gradient_flat)):
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.lr*gradient_flat[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return
    

class Momentum:
    def __init__(self,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.v=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[0 for x in range(len(gradient_flat))]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.v[i]=self.gamma*self.v[i]+self.lr*gradient_flat[i]
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.v[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return
    
    
class AdaGrad:
    def __init__(self,lr,epsilon=1e-06):
        self.lr=lr
        self.epsilon=epsilon
        self.s=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.s=[0 for x in range(len(gradient_flat))]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.s[i]=self.s[i]+gradient_flat[i]**2
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.lr*gradient_flat[i]/tf.sqrt(self.s[i]+self.epsilon))
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return
    

class RMSProp:
    def __init__(self,lr,gamma=0.9,epsilon=1e-06):
        self.lr=lr
        self.gamma=gamma
        self.epsilon=epsilon
        self.s=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.s=[0 for x in range(len(gradient_flat))]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.s[i]=self.gamma*self.s[i]+(1-self.gamma)*gradient_flat[i]**2
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.lr*gradient_flat[i]/tf.sqrt(self.s[i]+self.epsilon))
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
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
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.s=[0 for x in range(len(gradient_flat))]
            self.x=[0 for x in range(len(gradient_flat))]
            self.g=[x for x in range(len(gradient_flat))]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.s[i]=self.rho*self.s[i]+(1-self.rho)*gradient_flat[i]**2
            self.g[i]=tf.sqrt((self.x[i]+self.epsilon)/(self.s[i]+self.epsilon))*gradient_flat[i]
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
            self.x[i]=self.rho*self.x[i]+(1-self.rho)*self.g[i]**2
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
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
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[0 for x in range(len(gradient_flat))]
            self.s=[0 for x in range(len(gradient_flat))]
            self.v_=[x for x in range(len(gradient_flat))]
            self.s_=[x for x in range(len(gradient_flat))]
            self.g=[x for x in range(len(gradient_flat))]
            self.flag+=1
        for i in range(len(gradient_flat)):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i]
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2
            self.v_[i]=self.v[i]/(1-self.beta1**(t+1))
            self.s_[i]=self.s[i]/(1-self.beta2**(t+1))
            self.g[i]=self.lr*self.v_[i]/(tf.sqrt(self.s_[i])+self.epsilon)
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
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
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[0 for x in range(len(gradient_flat))]
            self.s=[0 for x in range(len(gradient_flat))]
            self.v_=[x for x in range(len(gradient_flat))]
            self.s_=[x for x in range(len(gradient_flat))]
            self.g=[x for x in range(len(gradient_flat))]
            self.m=[x for x in range(len(gradient_flat))]
            self.flag+=1
        for i in range(len(gradient_flat)):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i]
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2
            self.v_[i]=self.v[i]/(1-self.beta1**(t+1))
            self.s_[i]=self.s[i]/(1-self.beta2**(t+1))
            self.m[i]=(self.beta1*gradient_flat[i])/(1-self.beta1**(t+1))
            self.g[i]=self.lr*(self.m[i]+(1-self.beta1)*gradient_flat[i])/(tf.sqrt(self.s_[i])+self.epsilon)
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
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
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[0 for x in range(len(gradient_flat))]
            self.u=[0 for x in range(len(gradient_flat))]
            self.g=[x for x in range(len(gradient_flat))]
            self.flag+=1
        for i in range(len(gradient_flat)):
            self.v[i]=self.beta_1*self.v[i]+(1-self.beta_1)*gradient_flat[i]
            self.u[i]=tf.maximum(self.beta_2*self.u[i],tf.abs(gradient_flat[i]))
            self.g[i]=self.learning_rate/(1-self.beta_1**(t+1))*self.v[i]/(self.u[i]+self.epsilon)
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return


class AdamW:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-07,weight_decay=0.01):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.weight_decay=weight_decay
        self.v=[]
        self.s=[]
        self.v_=[]
        self.s_=[]
        self.g=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[0 for x in range(len(gradient_flat))]
            self.s=[0 for x in range(len(gradient_flat))]
            self.v_=[x for x in range(len(gradient_flat))]
            self.s_=[x for x in range(len(gradient_flat))]
            self.g=[x for x in range(len(gradient_flat))]
            self.flag+=1
        for i in range(len(gradient_flat)):
            gradient_flat[i]=gradient_flat[i]+self.weight_decay*parameter_flat[i]
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i]
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2
            self.v_[i]=self.v[i]/(1-self.beta1**(t+1))
            self.s_[i]=self.s[i]/(1-self.beta2**(t+1))
            self.g[i]=self.lr*self.v_[i]/(tf.sqrt(self.s_[i])+self.epsilon)
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return


class RAdam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-07):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.smoothing=2/(1-beta2)-1 
        self.v=[]
        self.s=[]
        self.v_=[]
        self.s_=[]
        self.g=[]
        self.step_size=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[0 for x in range(len(gradient_flat))]
            self.s=[0 for x in range(len(gradient_flat))]
            self.v_=[x for x in range(len(gradient_flat))]
            self.s_=[x for x in range(len(gradient_flat))]
            self.g=[x for x in range(len(gradient_flat))]
            self.step_size=[x for x in range(len(gradient_flat))]
            self.flag+=1
        for i in range(len(gradient_flat)):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i]
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2
            self.v_[i]=self.v[i]/(1-self.beta1**t)
            self.s_[i]=self.s[i]/(1-self.beta2**t)
            sma=self.smoothing-2*t*(self.beta2**t)/(1-self.beta2**t)
            if sma>=5:
                r=tf.math.sqrt((sma-4)*(sma-2)*self.smoothing/((self.smoothing-4)*(self.smoothing-2)*sma))
                self.g[i]=r*gradient_flat[i]/(tf.math.sqrt(self.s_[i])+self.epsilon)
                self.step_size[i]=-self.lr*r/(tf.math.sqrt(self.s_[i])+self.epsilon)
            else:
                self.g[i]=gradient_flat[i]
                self.step_size[i]=-self.lr/(tf.math.sqrt(self.s_[i])+self.epsilon)
            parameter_flat[i]=parameter_flat[i]+self.step_size[i]*self.v_[i]
        return nest.pack_sequence_as(parameter,parameter_flat)


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
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.n=[self.initial_accumulator_value for x in range(len(gradient_flat))]
            self.sigma=[0 for x in range(len(gradient_flat))]
            self.z=[0 for x in range(len(gradient_flat))]
            self.g=[x for x in range(len(gradient_flat))]
            self.flag+=1
        for i in range(len(gradient_flat)):
            prev_n=self.n[i]
            self.n[i]=self.n[i]+gradient_flat[i]**2
            self.sigma[i]=(self.n[i]**-self.learning_rate_power-prev_n**-self.learning_rate_power)/self.learning_rate
            self.z[i]=self.z[i]+gradient_flat[i]-self.sigma[i]*parameter_flat[i]
            if tf.abs(self.z[i])<self.l1_regularization_strength:
                state_ops.assign(parameter_flat[i],tf.zeros_like(self.z[i]))
            else:
                state_ops.assign(parameter_flat[i],(tf.sign(self.z[i])*self.l1_regularization_strength-self.z[i])/((self.beta+tf.sqrt(self.n[i]))/self.learning_rate+self.l2_regularization_strength))
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return


class AutoLR:
    def __init__(self,optimizer,initial_lr,min_lr,max_lr,factor):
        # initialize the optimizer to use
        self.optimizer=optimizer
        # initialize the learning rate parameters
        self.initial_lr=initial_lr
        self.min_lr=min_lr
        self.max_lr=max_lr
        self.factor=factor # this is the attribute that is used to change the learning rate
        # initialize the current learning rate and iteration counter
        self.current_lr=initial_lr
        self.iteration=0
    
    
    def opt(self,loss,parameter):
        # increment the iteration counter
        self.iteration+=1
        # compute the gradient of the loss with respect to the parameter
        gradient=tf.gradients(loss,parameter)
        if tf.math.is_nan(loss):
            # if loss is NaN, reduce learning rate by self.factor and reset iteration counter
            print("Loss is NaN. Reducing learning rate by factor.")
            new_lr=max(self.current_lr/self.factor,self.min_lr) # use self.factor here
            print("New learning rate:{}".format(new_lr))
            # update the learning rate of the custom optimizer
            self.optimizer.lr=new_lr
            self.optimizer.flag=0 # reset the flag to reinitialize the lists
        elif tf.math.is_inf(loss):
            # if loss is Inf, increase learning rate by self.factor and reset iteration counter
            print("Loss is Inf. Increasing learning rate by factor.")
            new_lr=min(self.current_lr*self.factor,self.max_lr) # use self.factor here
            print("New learning rate:{}".format(new_lr))
            # update the learning rate of the custom optimizer
            self.optimizer.lr=new_lr
            self.optimizer.flag=0 # reset the flag to reinitialize the lists
        else:
            # if loss is finite, update the parameter using the optimizer
            parameter=self.optimizer.opt(gradient,parameter,self.iteration)
        # update the current learning rate
        self.current_lr=self.optimizer.lr
        return parameter
