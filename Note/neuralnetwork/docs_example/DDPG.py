import tensorflow as tf
from Note import nn
import numpy as np
import gym


class actor(nn.Model):
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim, activation='tanh')
        self.action_bound=action_bound
    
    def __call__(self,x):
        x = self.dense1(x)
        return self.dense2(x)*self.action_bound


class critic(nn.Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self,x,a):
        cat=tf.concat([x,a],axis=1)
        x=self.dense1(cat)
        return self.dense2(x)


class DDPG(nn.RL):
    def __init__(self,hidden_dim,sigma,gamma,tau,actor_lr,critic_lr):
        super().__init__()
        self.env=gym.make('Pendulum-v0')
        state_dim=self.genv.observation_space.shape[0]
        action_dim=self.genv.action_space.shape[0]
        action_bound=self.genv.action_space.high[0]
        self.actor=actor(state_dim,hidden_dim,action_dim,action_bound)
        self.actor.detach()
        self.critic=critic(state_dim,hidden_dim,action_dim)
        self.critic.detach()
        self.target_actor=actor(state_dim,hidden_dim,action_dim,action_bound)
        self.target_actor.detach()
        self.target_critic=critic(state_dim,hidden_dim,action_dim)
        nn.assign_param(self.target_actor.param,self.actor_param.copy())
        nn.assign_param(self.target_critic.param,self.critic_param.copy())
        self.param=[self.actor.param,self.critic.param]
        self.sigma=sigma
        self.gamma=gamma
        self.tau=tau
    
    def noise(self):
        return np.random.normal(scale=self.sigma)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        next_q_value=self.target_critic.fp(next_s,self.target_actor.fp(next_s))
        q_target=r+self.gamma*next_q_value*(1-d)
        actor_loss=-tf.reduce_mean(self.critic.fp(s,self.actor.fp(s)))
        critic_loss=tf.reduce_mean((self.critic.fp(s,a)-q_target)**2)
        return -actor_loss+critic_loss
    
    def update_param(self):
        for target_param,param in zip(self.target_actor.param,self.actor.param):
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        for target_param,param in zip(self.target_critic.param,self.critic.param):
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        return