import tensorflow as tf
from Note import nn
from keras.models import Sequential
from keras import Model
import gym


class actor(Model):
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim, activation='tanh'))
        self.action_bound=action_bound
    
    def __call__(self,x):
        x = self.model(x)
        return x*self.action_bound


class critic(Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim+action_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim))
    
    def __call__(self,x,a):
        cat=tf.concat([x,a],axis=1)
        x=self.model(cat)
        return x


class DDPG(nn.RL):
    def __init__(self,hidden_dim,sigma,gamma,tau):
        super().__init__()
        self.env=gym.make('Pendulum-v1')
        state_dim=self.env.observation_space.shape[0]
        action_dim=self.env.action_space.shape[0]
        action_bound=self.env.action_space.high[0]
        self.actor=actor(state_dim,hidden_dim,action_dim,action_bound)
        self.critic=critic(state_dim,hidden_dim,action_dim)
        self.target_actor=actor(state_dim,hidden_dim,action_dim,action_bound)
        self.target_critic=critic(state_dim,hidden_dim,action_dim)
        nn.assign_param(self.target_actor.weights,self.actor.weights)
        nn.assign_param(self.target_critic.weights,self.critic.weights)
        self.param=[self.actor.weights,self.critic.weights]
        self.sigma=sigma
        self.gamma=gamma
        self.tau=tau
    
    def action(self,s):
        return self.actor(s)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        next_q_value=self.target_critic(next_s,self.target_actor(next_s))
        q_target=tf.cast(r,'float32')+self.gamma*next_q_value*(1-tf.cast(d,'float32'))
        actor_loss=-tf.reduce_mean(self.critic(s,self.actor(s)))
        critic_loss=tf.reduce_mean((self.critic(s,a)-q_target)**2)
        return [actor_loss,critic_loss]
    
    def update_param(self):
        for target_param,param in zip(self.target_actor.weights,self.actor.weights):
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        for target_param,param in zip(self.target_critic.weights,self.critic.weights):
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        return