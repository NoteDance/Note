import tensorflow as tf
from Note import nn
import gym


class actor(nn.Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self,x):
        x=self.dense1(x)
        return tf.nn.softmax(self.dense2(x))


class critic(nn.Model):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(1, hidden_dim)
    
    def __call__(self,x):
        x=self.dense1(x)
        return self.dense2(x)
    
    
class PPO(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps,alpha):
        super().__init__()
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.actor.detach()
        self.actor_old=actor(state_dim,hidden_dim,action_dim)
        self.actor_old.detach()
        nn.assign_param(self.actor_old.param,self.actor.param.copy())
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.alpha=alpha
        self.param=[self.actor.param,self.critic.param]
        self.env=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.actor_old(s)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        action_prob=tf.gather(self.actor(s),a,axis=1,batch_dims=1)
        action_prob_old=tf.gather(self.actor_old(s),a,axis=1,batch_dims=1)
        raito=action_prob/action_prob_old
        value=self.critic(s)
        value_tar=r+0.98*self.critic(next_s)*(1-tf.cast(d,'float32'))
        TD=value_tar-value
        sur1=raito*TD
        sur2=tf.clip_by_value(raito,clip_value_min=1-self.clip_eps,clip_value_max=1+self.clip_eps)*TD
        clip_loss=-tf.math.minimum(sur1,sur2)
        entropy=action_prob*tf.math.log(action_prob+1e-8)
        clip_loss=clip_loss-self.alpha*entropy
        return -tf.reduce_mean(clip_loss)+tf.reduce_mean((TD)**2)
    
    def update_param(self):
        nn.assign_param(self.nn.param, self.actor.param.copy())
        return
