import torch
from Note import nn
import numpy as np
import torch.nn.functional as F
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


class actor(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(actor,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class critic(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(critic,self).__init__()
        self.fc1=torch.nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)
    
    def forward(self,x,a):
        cat=torch.cat([x,a],dim=1)
        x=F.relu(self.fc1(cat))
        return self.fc2(x)


class DDPG(nn.RL_pytorch):
    def __init__(self,hidden_dim,sigma,gamma,tau):
        super().__init__()
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.env=MultiAgentEnv()
        state_dim=self.env.observation_space[0].shape[0]
        action_dim=self.env.action_space[0].n
        self.actor=[actor(state_dim,hidden_dim,action_dim).to(self.device) for _ in range(self.env.num_agents)]
        self.critic=[critic(state_dim,hidden_dim,action_dim).to(self.device) for _ in range(self.env.num_agents)]
        self.target_actor=[actor(state_dim,hidden_dim,action_dim).to(self.device) for _ in range(self.env.num_agents)]
        self.target_critic=[critic(state_dim,hidden_dim,action_dim).to(self.device) for _ in range(self.env.num_agents)]
        [self.target_actor[i].load_state_dict(self.actor[i].state_dict()) for i in range(self.env.num_agents)]
        [self.target_critic[i].load_state_dict(self.critic[i].state_dict()) for i in range(self.env.num_agents)]
        self.sigma=sigma
        self.gamma=gamma
        self.tau=tau
        self.param=[[self.actor[i].parameters(),self.critici[i].parameters()] for i in range(self.env.num_agents)]
    
    def action(self,s,i):
        return self.actor[i](s)
    
    def reward_done_func_ma(self,r,d):
        return torch.mean(r),all(d)
    
    def loss(self,s,a,next_s,r,d,i_agent):
        s=torch.tensor(s,dtype=torch.float).to(self.device)
        a=torch.tensor(a,dtype=torch.float).to(self.device)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device)
        r=torch.tensor(r,dtype=torch.float).to(self.device)
        d=torch.tensor(d,dtype=torch.float).to(self.device)
        next_q_value=self.target_critic(next_s,self.target_actor(next_s))
        q_target=r+self.gamma*next_q_value*(1-d)
        actor_loss=-torch.mean(self.critic(s[:,i_agent],self.actor(s[:,i_agent])))
        critic_loss=torch.mean(F.mse_loss(self.critic(s,a),q_target))
        return actor_loss+critic_loss
    
    def __call__(self,s,a,next_s,r,d):
        loss=0
        for i_agent in range(self.env.num_agents):
            loss+=self.loss(s,a,next_s,r,d,i_agent)
        return loss
    
    def update_param(self):
        for i in range(self.env.num_agents):
            for target_param,param in zip(self.target_actor[i].parameters(),self.actor[i].parameters()):
                target_param.data.copy_(target_param.data*(1.0-self.tau)+param.data*self.tau)
            for target_param,param in zip(self.target_critic[i].parameters(),self.critic[i].parameters()):
                target_param.data.copy_(target_param.data*(1.0-self.tau)+param.data*self.tau)
        return