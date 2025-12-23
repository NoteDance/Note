import tensorflow as tf
from Note import nn
from keras.models import Sequential
from keras import Model
import gym


class actor(Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim))
    
    def __call__(self,x):
        x=self.model(x)
        return tf.nn.softmax(x)


class critic(Model):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))
    
    def __call__(self,x):
        x=self.model(x)
        return x


class PPO(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps,alpha):
        super().__init__()
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.actor_old=actor(state_dim,hidden_dim,action_dim)
        nn.assign_param(self.actor_old.weights,self.actor.weights)
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.alpha=alpha
        self.batch_params={}
        self.batch_params['min']=None
        self.batch_params['max']=None
        self.batch_params['scale']=1.0
        self.batch_params['align']=None
        self.param=[self.actor.weights,self.critic.weights]
        self.env=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.actor_old(s)
    
    def window_size(self):
        return self.adjust_window_size()
    
    def window_size_func(self):
        return self.adjust_window_size()
    
    def adjust_func(self):
        if self.step_counter%777 or self.step_counter%self.update_steps==0:
            self.adjust(batch_params=self.batch_params)
            return
        self.adjust(batch_params=self.batch_params)
    
#    def adjust_func(self):
#        if self.step_counter%self.update_steps==0:
#            self.adjust(num_samples=7, target_noise=1e-3)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        action_prob=tf.gather(self.actor(s),a,axis=1,batch_dims=1)
        action_prob_old=tf.gather(self.actor_old(s),a,axis=1,batch_dims=1)
        ratio=action_prob/action_prob_old
        value=self.critic(s)
        value_tar=tf.cast(r,'float32')+0.98*self.critic(next_s)*(1-tf.cast(d,'float32'))
        TD=value_tar-value
        sur1=ratio*TD
        sur2=tf.clip_by_value(ratio,clip_value_min=1-self.clip_eps,clip_value_max=1+self.clip_eps)*TD
        clip_loss=-tf.math.minimum(sur1,sur2)
        entropy=action_prob*tf.math.log(action_prob+1e-8)
        clip_loss=clip_loss-self.alpha*entropy
        self.prioritized_replay.update(TD,ratio)
        return tf.reduce_mean(clip_loss)+tf.reduce_mean((TD)**2)
    
    def update_param(self):
        nn.assign_param(self.actor_old.weights, self.actor.weights)
        return
