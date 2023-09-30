import tensorflow as tf
from multiprocessing import Manager
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
        return parameter
    

class Momentum:
    def __init__(self,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        manager=Manager()
        self.v=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.v[i].assign(self.gamma*self.v[i]+self.lr*gradient_flat[i])
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.v[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter
    
    
class AdaGrad:
    def __init__(self,lr,epsilon=1e-06):
        self.lr=lr
        self.epsilon=epsilon
        manager=Manager()
        self.s=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.s=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.s[i].assign(self.s[i]+gradient_flat[i]**2)
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.lr*gradient_flat[i]/tf.sqrt(self.s[i]+self.epsilon))
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter
    

class RMSProp:
    def __init__(self,lr,gamma=0.9,epsilon=1e-06):
        self.lr=lr
        self.gamma=gamma
        self.epsilon=epsilon
        manager=Manager()
        self.s=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.s=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.s[i].assign(self.gamma*self.s[i]+(1-self.gamma)*gradient_flat[i]**2)
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.lr*gradient_flat[i]/tf.sqrt(self.s[i]+self.epsilon))
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


class AdaDelta:
    def __init__(self,lr,rho=0.95,epsilon=1e-05):
        self.lr=lr
        self.rho=rho
        self.epsilon=epsilon
        manager=Manager()
        self.s=manager.list()
        self.x=manager.list()
        self.g=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.s=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.x=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.g=[tf.Variable(x) for x in gradient_flat]
            self.flag=1
        for i in range(len(gradient_flat)):
            self.s[i].assign(self.rho*self.s[i]+(1-self.rho)*gradient_flat[i]**2)
            self.g[i].assign(tf.sqrt((self.x[i]+self.epsilon)/(self.s[i]+self.epsilon))*gradient_flat[i])
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
            self.x[i].assign(self.rho*self.x[i]+(1-self.rho)*self.g[i]**2)
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-07):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        manager=Manager()
        self.v=manager.list()
        self.s=manager.list()
        self.v_=manager.list()
        self.s_=manager.list()
        self.g=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.s=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.v_=[tf.Variable(x) for x in gradient_flat]
            self.s_=[tf.Variable(x) for x in gradient_flat]
            self.g=[tf.Variable(x) for x in gradient_flat]
            self.flag+=1
        for i in range(len(gradient_flat)):
            if t.dtype!=gradient_flat[i].dtype:
                t=tf.cast(t,gradient_flat[i].dtype)
            self.v[i].assign(self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i])
            self.s[i].assign(self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2)
            self.v_[i].assign(self.v[i]/(1-self.beta1**(t+1)))
            self.s_[i].assign(self.s[i]/(1-self.beta2**(t+1)))
            self.g[i].assign(self.lr*self.v_[i]/(tf.sqrt(self.s_[i])+self.epsilon))
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


class Nadam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-07):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        manager=Manager()
        self.v=manager.list()
        self.s=manager.list()
        self.v_=manager.list()
        self.s_=manager.list()
        self.g=manager.list()
        self.m=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.s=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.v_=[tf.Variable(x) for x in gradient_flat]
            self.s_=[tf.Variable(x) for x in gradient_flat]
            self.g=[tf.Variable(x) for x in gradient_flat]
            self.m=[tf.Variable(x) for x in gradient_flat]
            self.flag+=1
        for i in range(len(gradient_flat)):
            if t.dtype!=gradient_flat[i].dtype:
                t=tf.cast(t,gradient_flat[i].dtype)
            self.v[i].assign(self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i])
            self.s[i].assign(self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2)
            self.v_[i].assign(self.v[i]/(1-self.beta1**(t+1)))
            self.s_[i].assign(self.s[i]/(1-self.beta2**(t+1)))
            self.m[i].assign((self.beta1*gradient_flat[i])/(1-self.beta1**(t+1)))
            self.g[i].assign(self.lr*(self.m[i]+(1-self.beta1)*gradient_flat[i])/(tf.sqrt(self.s_[i])+self.epsilon))
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


class AdaMax:
    def __init__(self,learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07):
        self.learning_rate=learning_rate
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon
        manager=Manager()
        self.v=manager.list()
        self.u=manager.list()
        self.g=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.u=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.g=[tf.Variable(x) for x in gradient_flat]
            self.flag+=1
        for i in range(len(gradient_flat)):
            if t.dtype!=gradient_flat[i].dtype:
                t=tf.cast(t,gradient_flat[i].dtype)
            self.v[i].assign(self.beta_1*self.v[i]+(1-self.beta_1)*gradient_flat[i])
            self.u[i].assign(tf.maximum(self.beta_2*self.u[i],tf.abs(gradient_flat[i])))
            self.g[i].assign(self.learning_rate/(1-self.beta_1**(t+1))*self.v[i]/(self.u[i]+self.epsilon))
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


class AdamW:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-07,weight_decay=0.01):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.weight_decay=weight_decay
        manager=Manager()
        self.v=manager.list()
        self.s=manager.list()
        self.v_=manager.list()
        self.s_=manager.list()
        self.g=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.s=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.v_=[tf.Variable(x) for x in gradient_flat]
            self.s_=[tf.Variable(x) for x in gradient_flat]
            self.g=[tf.Variable(x) for x in gradient_flat]
            self.flag+=1
        for i in range(len(gradient_flat)):
            if t.dtype!=gradient_flat[i].dtype:
                t=tf.cast(t,gradient_flat[i].dtype)
            gradient_flat[i]=gradient_flat[i]+self.weight_decay*parameter_flat[i]
            self.v[i].assign(self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i])
            self.s[i].assign(self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2)
            self.v_[i].assign(self.v[i]/(1-self.beta1**(t+1)))
            self.s_[i].assign(self.s[i]/(1-self.beta2**(t+1)))
            self.g[i].assign(self.lr*self.v_[i]/(tf.sqrt(self.s_[i])+self.epsilon))
            state_ops.assign(parameter_flat[i],parameter_flat[i]-self.g[i])
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


class RAdam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-07):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.smoothing=2/(1-beta2)-1 
        manager=Manager()
        self.v=manager.list()
        self.s=manager.list()
        self.v_=manager.list()
        self.s_=manager.list()
        self.g=manager.list()
        self.step_size=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.v=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.s=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.v_=[tf.Variable(x) for x in gradient_flat]
            self.s_=[tf.Variable(x) for x in gradient_flat]
            self.g=[tf.Variable(x) for x in gradient_flat]
            self.step_size=[tf.Variable(x) for x in gradient_flat]
            self.flag+=1
        for i in range(len(gradient_flat)):
            if t.dtype!=gradient_flat[i].dtype:
                t=tf.cast(t,gradient_flat[i].dtype)
            self.v[i].assign(self.beta1*self.v[i]+(1-self.beta1)*gradient_flat[i])
            self.s[i].assign(self.beta2*self.s[i]+(1-self.beta2)*gradient_flat[i]**2)
            self.v_[i].assign(self.v[i]/(1-self.beta1**t))
            self.s_[i].assign(self.s[i]/(1-self.beta2**t))
            sma=self.smoothing-2*t*(self.beta2**t)/(1-self.beta2**t)
            if sma>=5:
                r=tf.math.sqrt((sma-4)*(sma-2)*self.smoothing/((self.smoothing-4)*(self.smoothing-2)*sma))
                self.g[i].assign(r*gradient_flat[i]/(tf.math.sqrt(self.s_[i])+self.epsilon))
                self.step_size[i].assign(-self.lr*r/(tf.math.sqrt(self.s_[i])+self.epsilon))
            else:
                self.g[i].assign(gradient_flat[i])
                self.step_size[i].assign(-self.lr/(tf.math.sqrt(self.s_[i])+self.epsilon))
            state_ops.assign(parameter_flat[i],parameter_flat[i]+self.step_size[i]*self.v_[i])
            parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


class Ftrl:
    def __init__(self,learning_rate=0.001,learning_rate_power=-0.5,initial_accumulator_value=0.1,l1_regularization_strength=0.0,l2_regularization_strength=0.0,l2_shrinkage_regularization_strength=0.0,beta=0.0):
        self.learning_rate=learning_rate
        self.learning_rate_power=learning_rate_power
        self.initial_accumulator_value=initial_accumulator_value
        self.l1_regularization_strength=l1_regularization_strength
        self.l2_regularization_strength=l2_regularization_strength
        self.l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength
        self.beta=beta
        manager=Manager()
        self.n=manager.list()
        self.sigma=manager.list()
        self.z=manager.list()
        self.g=manager.list()
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.n=[tf.Variable(self.initial_accumulator_value*tf.ones_like(x),dtype=x.dtype.name) for x in range(len(gradient_flat))]
            self.sigma=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.z=[tf.Variable(tf.zeros_like(x),dtype=x.dtype.name) for x in gradient_flat]
            self.g=[tf.Variable(x) for x in gradient_flat]
            self.flag+=1
        for i in range(len(gradient_flat)):
            prev_n=self.n[i]
            self.n[i].assign(self.n[i]+gradient_flat[i]**2)
            self.sigma[i].assign((self.n[i]**-self.learning_rate_power-prev_n**-self.learning_rate_power)/self.learning_rate)
            self.z[i].assign(self.z[i]+gradient_flat[i]-self.sigma[i]*parameter_flat[i])
            if tf.abs(self.z[i])<self.l1_regularization_strength:
                state_ops.assign(parameter_flat[i],tf.zeros_like(self.z[i]))
            else:
                state_ops.assign(parameter_flat[i],(tf.sign(self.z[i])*self.l1_regularization_strength-self.z[i])/((self.beta+tf.sqrt(self.n[i]))/self.learning_rate+self.l2_regularization_strength))
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


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


# Define a LookAhead optimizer class
class LookAhead:
    # Initialization method, takes an inner optimizer, sync period, slow step size and other parameters
    def __init__(self,optimizer,sync_period=6,slow_step_size=0.5):
        # Save attributes
        self.optimizer=optimizer
        self.sync_period=sync_period
        self.slow_step_size=slow_step_size
        # Initialize a dictionary for slow weights
        manager=Manager()
        self.slow_weights=manager.dict()
    

    # Define an opt method, used to apply gradients
    def opt(self,gradient,parameter,t):
        # Call the opt method of the inner optimizer, update the fast weights
        self.optimizer.opt(gradient,parameter,t)
        # Get the current iteration number
        local_step=t
        # Determine whether to sync the slow weights and the fast weights
        sync_cond=local_step%self.sync_period==0
        # If sync is needed, update all the slow weights and assign them to the fast weights
        if sync_cond:
            for var in parameter:
                # If the variable does not have a corresponding slow weight, create one and initialize it to the value of the variable
                if var not in self.slow_weights:
                    self.slow_weights[var]=var.copy()
                # Calculate the new slow weight and assign it to the slow weight and the fast weight
                new_slow_var=self.slow_weights[var]+self.slow_step_size*(var-self.slow_weights[var])
                self.slow_weights[var]=new_slow_var
                var[:]=new_slow_var
        return parameter
