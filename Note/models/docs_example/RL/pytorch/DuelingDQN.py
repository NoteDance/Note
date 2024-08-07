import torch
from Note import nn
import gym
import torch.nn.functional as F


class VAnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(VAnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc_A=torch.nn.Linear(hidden_dim,action_dim)
        self.fc_V=torch.nn.Linear(hidden_dim,1)
    
    def forward(self,x):
        A=self.fc_A(F.relu(self.fc1(x)))
        V=self.fc_V(F.relu(self.fc1(x)))
        Q=V+A-A.mean(1).view(-1,1)
        return Q
    
    
class DuelingDQN(nn.RL_pytorch):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.va_net=VAnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.target_q_net=VAnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.optimizer=torch.optim.Adam(self.nn.parameters(),lr=2e-3)
        self.genv=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.va_net(s)
    
    def loss(self,s,a,next_s,r,d):
        s=torch.tensor(s,dtype=torch.float).to(self.device)
        a=torch.tensor(a,dtype=torch.int64).view(-1,1).to(self.device)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device)
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(self.device)
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(self.device)
        q_value=self.va_net(s).gather(1,a)
        next_q_value=self.target_q_net(next_s).max(1)[0].view(-1,1)
        target=r+0.98*next_q_value*(1-d)
        return F.mse_loss(q_value,target)
    
    def update_param(self):
        self.target_q_net.load_state_dict(self.nn.state_dict())
        return