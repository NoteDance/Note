import tensorflow as tf
from Note import nn
from keras.models import Sequential
from keras import Model
import gym


class Qnet(Model):
    def __init__(self,state_dim, hidden_dim, action_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim))
    
    def __call__(self,x):
        x = self.model(x)
        return x


class Controller(Model):
    def __init__(self, hidden=32, temp=10.0):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')
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
        self.param=self.q_net.weights
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.q_net(s)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        TD=(q_value-target)
        self.prioritized_replay.update(TD)
        return tf.reduce_mean(TD**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.weights,self.param)
        return


class DQN_(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,processes):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.q_net.weights
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.q_net(s)
    
    def window_size(self,p):
        return self.adjust_window_size(p)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        TD=(q_value-target)
        self.prioritized_replay.update(TD)
        return tf.reduce_mean(TD**2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.weights,self.param)
        return


class _DQN(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,processes,temp=10.0):
        super().__init__()
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.temp=temp
        self.param=self.q_net.weights
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.q_net(s)
    
    def window_size(self):
        td_score = tf.reduce_sum(self.prioritized_replay.TD)
        weights = tf.pow(td_score + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([td_score, ess, len(self.prioritized_replay.TD)], (1,3))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        return self.controller(features)
    
    def window_size_fn(self):
        td_score = tf.reduce_sum(self.prioritized_replay.TD)
        scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
        weights = tf.pow(scores + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([td_score, ess, len(self.prioritized_replay.TD)], (1,3))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        return self.controller(features)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.q_net(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net(next_s),axis=1)
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32'))
        TD=(q_value-target)
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
        controller_loss = tf.reduce_mean(m * td_score)
        self.prioritized_replay.update(TD)
        return tf.reduce_mean(TD**2)+controller_loss
    
    def update_param(self):
        nn.assign_param(self.target_q_net.weights,self.param)
        return