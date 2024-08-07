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
    
    
class DQN(nn.RL_pytorch):
    def __init__(self,state_dim,hidden_dim,action_dim):
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.q_net=Qnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.param=self.q_net.parameters()
        self.env=gym.make('CartPole-v0') #create environment
    
    def action(self,s):
        return self.q_net(s)
    
    def __call__(self,s,a,next_s,r,d): #loss function,kernel uses it to calculate loss.
        s=torch.tensor(s,dtype=torch.float).to(self.device)
        a=torch.tensor(a,dtype=torch.int64).view(-1,1).to(self.device)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device)
        r=torch.tensor(r,dtype=torch.float).to(self.device)
        d=torch.tensor(d,dtype=torch.float).to(self.device)
        q_value=self.q_net(s).gather(1,a)
        next_q_value=self.target_q_net(next_s).max(1)[0]
        target=r+0.98*next_q_value*(1-d)
        return F.mse_loss(q_value,target)
    
    def update_param(self): #update function,kernel uses it to update parameter.
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        return
