import torch
from Note import nn
import gym
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)
    
    
class DoubleDQN(nn.RL_pytorch):
    def __init__(self,state_dim,hidden_dim,action_dim,processes):
        super().__init__()
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.q_net=Qnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.param=self.q_net.parameters()
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.q_net(s)
    
    def __call__(self,s,a,next_s,r,d):
        s=torch.tensor(s,dtype=torch.float).to(self.device)
        a=torch.tensor(a,dtype=torch.int64).view(-1,1).to(self.device)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device)
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(self.device)
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(self.device)
        q_value=self.q_net(s).gather(1,a)
        max_action=self.q_net(next_s).max(1)[1].view(-1,1)
        next_q_value=self.target_q_net(next_s).gather(1,max_action)
        target=r+0.98*next_q_value*(1-d)
        return F.mse_loss(q_value,target)
    
    def update_param(self):
        self.target_q_net.load_state_dict(self.nn.state_dict())
        return