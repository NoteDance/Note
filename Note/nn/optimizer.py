import tensorflow as tf
import numpy as np
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
            state_ops.assign(parameter_flat[i],parameter_flat[i]+self.step_size[i]*self.v_[i])
            parameter=nest.pack_sequence_as(parameter,parameter_flat)
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
        return


# Define a LookAhead optimizer class
class LookAhead:
    # Initialization method, takes an inner optimizer, sync period, slow step size and other parameters
    def __init__(self,optimizer,sync_period=6,slow_step_size=0.5):
        # Save attributes
        self.optimizer=optimizer
        self.sync_period=sync_period
        self.slow_step_size=slow_step_size
        # Initialize a dictionary for slow weights
        self.slow_weights=dict()
    

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
        return


# Define a Ranger optimizer class
class Ranger:
    # Initialization method, receive an internal optimizer (default is RAdam), sync period, slow step size and other parameters
    def __init__(self,optimizer=RAdam(),sync_period=6,slow_step_size=0.5,**kwargs):
        # Save attributes
        # Initialize slow weight dictionary
        self.slow_weights=dict()
        # Initialize other parameters, such as adaptive gradient clipping, positive negative momentum, norm loss, etc.
        self.adaptive_grad_clip=kwargs.get("adaptive_grad_clip",True)
        self.positive_negative_momentum=kwargs.get("positive_negative_momentum",True)
        self.norm_loss=kwargs.get("norm_loss",True)
        # Initialize linear learning rate warmup parameters, such as warmup steps and initial learning rate
        self.warmup_steps=kwargs.get("warmup_steps",0)
        self.init_lr=kwargs.get("init_lr",0.0)
        # Initialize explore-exploit learning rate schedule parameters, such as maximum learning rate and minimum learning rate
        self.max_lr=kwargs.get("max_lr",0.1)
        self.min_lr=kwargs.get("min_lr",0.01)
        # Initialize LookAhead parameters, such as whether to enable and sync period
        # If LookAhead is enabled, create an instance of the LookAhead class and pass the internal optimizer and sync period
        self.lookahead=LookAhead(optimizer,sync_period=sync_period,slow_step_size=slow_step_size)
    
    
    # Define a clip_grad method for adaptive gradient clipping
    def clip_grad(self,gradient):
        # Calculate the maximum and minimum values of the gradient according to the formula in the paper
        grad_max=tf.reduce_max(gradient)
        grad_min=tf.reduce_min(gradient)
        grad_max_abs=tf.maximum(tf.abs(grad_max),1e-12)
        grad_min_abs=tf.maximum(tf.abs(grad_min),1e-12)
        clip_val_max=grad_max/grad_max_abs*tf.minimum(grad_max_abs,10*grad_min_abs+1e-3)
        clip_val_min=grad_min/grad_min_abs*tf.minimum(grad_min_abs,10*grad_max_abs+1e-3)
        # Limit the gradient between the maximum and minimum values
        gradient=tf.clip_by_value(gradient,clip_val_min,clip_val_max)
        return gradient
    

    # Define an adjust_grad method for positive negative momentum adjustment
    def adjust_grad(self,gradient):
        # Calculate the positive and negative parts of the gradient according to the formula in the paper
        grad_pos=tf.maximum(gradient,0.0)
        grad_neg=tf.minimum(gradient,0.0)
        # Calculate the positive and negative momentum of the gradient according to the formula in the paper
        mom_pos=self.optimizer.beta1*grad_pos+(1-self.optimizer.beta1)*grad_neg
        mom_neg=self.optimizer.beta1*grad_neg + (1-self.optimizer.beta1)*grad_pos
        # Calculate the adjustment coefficient of the gradient according to the formula in the paper
        coef_pos=tf.where(grad_pos>0,1.0+mom_pos,1.0)
        coef_neg=tf.where(grad_neg<0,1.0+mom_neg,1.0)
        # Multiply the gradient by the adjustment coefficient
        gradient=gradient*coef_pos*coef_neg
        return gradient
    

    # Define a penalize_grad method for norm loss penalty
    def penalize_grad(self,gradient):
        # Calculate the norm of the gradient according to the formula in the paper
        grad_norm=tf.norm(gradient)
        # Calculate the penalty coefficient of the gradient according to the formula in the paper
        penalty=tf.exp(grad_norm)-1
        # Multiply the gradient by the penalty coefficient
        gradient=gradient*penalty
        return gradient
    

    # Define an adjust_lr method for linear learning rate warmup
    def adjust_lr(self,local_step):
        # Calculate the current learning rate according to the formula in the paper
        lr_ratio=(self.optimizer.lr-self.init_lr)/(self.warmup_steps-1)
        current_lr=self.init_lr+lr_ratio*local_step
        # Assign the current learning rate to the lr attribute of the internal optimizer
        self.optimizer.lr=current_lr
    

    # Define an explore_exploit_lr method for explore-exploit learning rate schedule
    def explore_exploit_lr(self,local_step):
        # Calculate the current learning rate according to the formula in the paper
        lr_ratio=(self.max_lr-self.min_lr)/(self.max_lr+self.min_lr)
        current_lr=(self.max_lr+self.min_lr)/2+(self.max_lr-self.min_lr)/2*tf.math.cos(np.pi*local_step/lr_ratio)
        # Assign the current learning rate to the lr attribute of the internal optimizer
        self.optimizer.lr=current_lr
    
    
    # Define a softplus method for Softplus transformation
    def softplus(self,x):
        # Calculate the value after Softplus transformation according to the formula in the paper
        return tf.math.log(1+tf.math.exp(x))
    

    # Define a normalize_grad method for gradient normalization
    def normalize_grad(self,gradient):
        # Calculate the mean and variance of the gradient according to the formula in the paper
        grad_mean=tf.reduce_mean(gradient)
        grad_var=tf.reduce_mean(tf.square(gradient-grad_mean))
        # Calculate the value after gradient normalization according to the formula in the paper
        gradient=(gradient-grad_mean)/(self.softplus(grad_var)+self.optimizer.epsilon)
        return gradient
    

    # Define an opt method for applying gradients
    def opt(self,gradient,parameter,t):
        # If adaptive gradient clipping is enabled, clip the gradient
        if self.adaptive_grad_clip:
            gradient=self.clip_grad(gradient)
        # If positive negative momentum is enabled, adjust the gradient
        if self.positive_negative_momentum:
            gradient=self.adjust_grad(gradient)
        # If norm loss is enabled, penalize the gradient
        if self.norm_loss:
            gradient=self.penalize_grad(gradient)
        # If gradient normalization is enabled, normalize the gradient
        if self.normalize_grad:
            gradient=self.normalize_grad(gradient)
        self.lookahead.opt(gradient,parameter,t)
        # If linear learning rate warmup is enabled, adjust the learning rate according to the current iteration number
        if self.warmup_steps>0:
            self.adjust_lr(t)
        # If explore-exploit learning rate schedule is enabled, adjust the learning rate according to the current iteration number
        if self.max_lr>self.min_lr:
            self.explore_exploit_lr(t)
        return
