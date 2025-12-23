import tensorflow as tf
from Note import nn
from keras.models import Sequential
from keras import Model
import gym


class Qnet(Model):
    def __init__(self,state_dim, hidden_dim, action_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim))
    
    def __call__(self,x):
        x = self.model(x)
        return x
    
    
class DQN(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.q_net.weights
        self.env=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.q_net(s)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        TD=(q_value-target)
        self.prioritized_replay.update(TD)
        return tf.reduce_mean(TD**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.weights,self.param)
        return


class DQN_(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.q_net.weights
        self.batch_params={}
        self.batch_params['min']=None
        self.batch_params['max']=None
        self.batch_params['scale']=1.0
        self.batch_params['align']=None
        self.env=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.q_net(s)
    
    def window_size(self):
        return self.adjust_window_size()
    
    def window_size_func(self):
        return self.adjust_window_size()
    
    def adjust_func(self):
        if self.step_counter%777 or self.step_counter%self.update_steps==0:
            self.adjust(batch_params=self.batch_params)
            return
        self.adjust(batch_params=self.batch_params)
    
#    def adjust_func(self):
#        if self.step_counter%self.update_steps==0:
#            self.adjust(num_samples=7, target_noise=1e-3)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        TD=(q_value-target)
        self.prioritized_replay.update(TD)
        return tf.reduce_mean(TD**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.weights,self.param)
        return
