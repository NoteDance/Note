import tensorflow as tf
from Note import nn
import numpy as np
import gym
from gym import spaces


class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, state_size=4, action_size=2, done_threshold=10):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.done_threshold = done_threshold

        # Define action and observation space
        self.action_space = [spaces.Discrete(action_size) for _ in range(num_agents)]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32) for _ in range(num_agents)]

        self.reset()

    def reset(self):
        self.states = [np.random.rand(self.state_size) for _ in range(self.num_agents)]
        return self.states

    def step(self, actions):
        rewards = []
        next_states = []
        done = [False, False]

        for i in range(self.num_agents):
            # Example transition logic
            next_state = self.states[i] + np.random.randn(self.state_size) * 0.1
            reward = -np.sum(np.square(actions[i] - 0.5))  # Example reward function
            rewards.append(reward)
            next_states.append(next_state)

            # Check if any agent's state exceeds the done threshold
            if np.any(next_state > self.done_threshold):
                done = [True, True]

        self.states = next_states
        return next_states, rewards, done, {}


class actor(nn.Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim, activation='tanh')
    
    def __call__(self,x):
        x = self.dense1(x)
        return self.dense2(x)


class critic(nn.Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim+action_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self,x,a):
        cat=tf.concat([x,a],axis=1)
        x=self.dense1(cat)
        return self.dense2(x)


class DDPG(nn.RL):
    def __init__(self,hidden_dim,sigma,gamma,tau,processes):
        super().__init__()
        self.env=[MultiAgentEnv() for _ in range(processes)]
        state_dim=self.env.observation_space[0].shape[0]
        action_dim=self.env.action_space[0].n
        self.actor=[actor(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        [self.actor[i].detach() for i in range(self.env.num_agents)]
        self.critic=[critic(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        [self.critic[i].detach() for i in range(self.env.num_agents)]
        self.target_actor=[actor(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        [self.target_actor[i].detach() for i in range(self.env.num_agents)]
        self.target_critic=[critic(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        [self.target_critic[i].detach() for i in range(self.env.num_agents)]
        [nn.assign_param(self.target_actor[i].param,self.actor[i].param) for i in range(self.env.num_agents)]
        [nn.assign_param(self.target_critic[i].param,self.critic[i].param) for i in range(self.env.num_agents)]
        self.param=[[self.actor[i].param,self.critic[i].param] for i in range(self.env.num_agents)]
        self.sigma=sigma
        self.gamma=gamma
        self.tau=tau
    
    def action(self,s,i):
        return self.actor[i](s)
    
    def reward_done_func_ma(self,r,d):
        return tf.reduce_mean(r),all(d)
    
    def loss(self,s,a,next_s,r,d,i_agent):
        next_q_value=self.target_critic[i_agent](next_s,self.target_actor[i_agent](next_s))
        q_target=tf.cast(r,'float32')+self.gamma*next_q_value*(1-tf.cast(d,'float32'))
        actor_loss=-tf.reduce_mean(self.critic[i_agent](s[:,i_agent],self.actor[i_agent](s[:,i_agent])))
        critic_loss=tf.reduce_mean(tf.reduce_mean((self.critic[i_agent](s,a)-q_target)**2))
        return actor_loss+critic_loss
    
    def __call__(self,s,a,next_s,r,d):
        loss=0
        for i_agent in range(self.env.num_agents):
            loss+=self.loss(s,a,next_s,r,d,i_agent)
        return loss
    
    def update_param(self):
        for i in range(self.env.num_agents):
            for target_param,param in zip(self.target_actor[i].param,self.actor[i].param):
                target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
            for target_param,param in zip(self.target_critic[i].param,self.critic[i].param):
                target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        return