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


class Controller(nn.Model):
    def __init__(self, hidden=32, temp=10.0):
        super().__init__()
        self.fc1 = nn.dense(hidden, 3, activation='relu')
        self.fc2 = nn.dense(1, hidden, activation='sigmoid')
        self.max_w = None
        self.temp = temp

    def __call__(self, features):
        x = self.fc1(features)
        alpha = self.fc2(x)
        w = alpha * self.max_w
        return tf.squeeze(w, axis=-1)
    
    
class DQN(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,processes):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.q_net.param
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.q_net(s)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        self.prioritized_replay.update(target)
        return tf.reduce_mean((q_value-target)**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param,self.param)
        return


class DQN_(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,processes):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.q_net.param
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.q_net(s)
    
    def window_size(self,p):
        return self.adjust_window_size(p)
    
    def window_size_fn(self,p):
        return self.adjust_window_size(p)
    
    def batch_size_fn(self):
        if self.batch_counter%777:
            return self.adjust_batch_size()
        return self.adjust_batch_size()
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        self.prioritized_replay.update(target)
        return tf.reduce_mean((q_value-target)**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param,self.param)
        return


class _DQN(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,processes,temp=10.0):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.temp=temp
        self.param=self.q_net.param
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.q_net(s)
    
    def window_size(self,p):
        td_score = tf.reduce_sum(self.prioritized_replay.TD_list[p])
        weights = tf.pow(td_score + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([td_score, ess, len(self.prioritized_replay.TD)], (1,3))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        return self.controller(features)
    
    def window_size_fn(self,p):
        td_score = tf.reduce_sum(self.prioritized_replay.TD_list[p])
        weights = tf.pow(td_score + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([td_score, ess, len(self.prioritized_replay.TD)], (1,3))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        return self.controller(features)
    
    def batch_size_fn(self):
        if self.batch_counter%777:
            return self.adjust_batch_size()
        return self.adjust_batch_size()
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        self.controller.max_w = len(self.prioritized_replay.TD)
        td_score = tf.reduce_sum(self.prioritized_replay.TD)
        weights = tf.pow(td_score + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([td_score, ess, len(self.prioritized_replay.TD)], (1,3))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        w = self.controller(features)
        idx = tf.cast(tf.range(len(self.prioritized_replay.ratio), w.dtype))
        m = tf.sigmoid((idx - w) / self.temp)
        controller_loss = -tf.reduce_mean(m * td_score)
        self.prioritized_replay.update(target)
        return tf.reduce_mean((q_value-target)**2)+controller_loss
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param,self.param)
        return