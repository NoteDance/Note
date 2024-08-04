import tensorflow as tf
from Note import nn
import numpy as np
import random


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
        self.dense1 = nn.dense(hidden_dim, state_dim+action_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self,x,a):
        cat=tf.concat([x,a],axis=1)
        x=self.dense1(cat)
        return self.dense2(x)


class WorldEnv:
    def __init__(self):
        self.distance_threshold = 0.15
        self.action_bound = 1
    
    def reset(self,seed):
        self.goal = np.array([4+ + random.uniform(-0.5, 0.5), 4 + random.uniform(-0.5, 0.5)])
        self.state = np.array([0, 0])
        self.count = 0
        return self.state
    
    def step(self, action):
        action = np.clip(action, -self.action_bound, self.action_bound)
        x = max(0, min(5, self.state[0] + action[0]))
        y = max(0, min(5, self.state[1] + action[1]))
        self.state = np.array([x, y])
        self.count += 1
        
        dis = np.sqrt(np.sum(np.square(self.state - self.goal)))
        reward = -1.0 if dis > self.distance_threshold else 0
        if dis <= self.distance_threshold or self.count == 50:
            done = True
        else:
            done = False
        
        return self.state, reward, done, None


class DDPG(nn.RL):
    def __init__(self,hidden_dim,sigma,gamma,tau):
        super().__init__()
        self.env=WorldEnv()
        state_dim=4
        action_dim=2
        action_bound=1
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
    
    def action(self,s):
        return self.actor(s)
    
    def reward_done_func(self, next_state, goal):
        dis = np.sqrt(np.sum(np.square(next_state - goal)))
        reward = -1.0 if dis > 0.15 else 0
        done = False if dis > 0.15 else True
        return reward, done
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        next_q_value=self.target_critic(next_s,self.target_actor(next_s))
        q_target=tf.cast(r,'float32')+self.gamma*next_q_value*(1-tf.cast(d,'float32'))
        actor_loss=-tf.reduce_mean(self.critic(s,self.actor(s)))
        critic_loss=tf.reduce_mean((self.critic(s,a)-q_target)**2)
        return -actor_loss+critic_loss
    
    def update_param(self):
        for target_param,param in zip(self.target_actor.param,self.actor.param):
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        for target_param,param in zip(self.target_critic.param,self.critic.param):
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        return