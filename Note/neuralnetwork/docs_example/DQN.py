import tensorflow as tf
from Note import nn
import gym


class Qnet(nn.Model):
    def __init__(self,state_dim, hidden_dim, action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self,x):
        x = self.dense2(self.dense1(x))
        return x
    
    
class DQN(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.q_net.detach()
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.q_net.param
        self.env=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.q_net(s)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        return tf.reduce_mean((q_value-target)**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param,self.param)
        return