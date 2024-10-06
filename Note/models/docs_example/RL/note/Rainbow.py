import tensorflow as tf
import numpy as np
from Note import nn
import gym


class NoisyLinear:
    def __init__(self, out_dim, in_dim, sigma_init=0.017):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.mu_w = nn.initializer([in_dim, out_dim], ['uniform',-1/np.sqrt(in_dim),1/np.sqrt(in_dim)])
        self.sigma_w = nn.Parameter(tf.ones([in_dim, out_dim])*sigma_init)
        self.mu_b = nn.initializer([out_dim], ['uniform',-1/np.sqrt(in_dim),1/np.sqrt(in_dim)])
        self.sigma_b = nn.Parameter(tf.ones([out_dim])*sigma_init)
        
    def __call__(self, x):
        epsilon_in = self._scale_noise(self.in_dim)
        epsilon_out = self._scale_noise(self.out_dim)
        
        w_noise = tf.multiply(self.sigma_w, tf.einsum('i,j->ij', epsilon_in, epsilon_out))
        b_noise = tf.multiply(self.sigma_b, epsilon_out)
        
        return tf.matmul(x, self.mu_w + w_noise) + (self.mu_b + b_noise)
    
    def _scale_noise(self, size):
        x = tf.random.normal([size])
        return tf.sign(x) * tf.sqrt(tf.abs(x))


class VAnet(nn.Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.fc1=NoisyLinear(hidden_dim,state_dim)
        self.fc_A=NoisyLinear(action_dim,hidden_dim)
        self.fc_V=NoisyLinear(1,hidden_dim)
    
    def __call__(self,x):
        A=self.fc_A(tf.nn.relu(self.fc1(x)))
        V=self.fc_V(tf.nn.relu(self.fc1(x)))
        Q=V+A-tf.expand_dims(tf.reduce_mean(A,axis=1),axis=1)
        return Q
    
    
class Rainbow(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.va_net=VAnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=VAnet(state_dim,hidden_dim,action_dim)
        self.param=self.va_net.param
        self.genv=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.va_net(s)
    
    def loss(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.va_net(s),a,axis=1,batch_dims=1)
        max_action=tf.expand_dims(tf.argmax(self.va_net(s),axis=1),axis=1)
        next_q_value=tf.gather(self.target_q_net(next_s),max_action,axis=1,batch_dims=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        TD=(q_value-target)
        self.prioritized_replay.update_TD(TD)
        return tf.reduce_mean(TD**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param,self.param)
        return