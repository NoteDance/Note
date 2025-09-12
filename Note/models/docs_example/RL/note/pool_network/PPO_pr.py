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


class Controller(nn.Model):
    def __init__(self, hidden=32, temp=10.0):
        super().__init__()
        self.fc1 = nn.dense(hidden, 4, activation='relu')
        self.fc2 = nn.dense(1, hidden, activation='sigmoid')
        self.max_w = None
        self.temp = temp

    def __call__(self, features):
        x = self.fc1(features)
        alpha = self.fc2(x)
        w = alpha * self.max_w
        return tf.squeeze(w, axis=-1)


class PPO(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps,alpha,processes):
        super().__init__()
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.actor_old=actor(state_dim,hidden_dim,action_dim)
        nn.assign_param(self.actor_old.param,self.actor.param)
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.alpha=alpha
        self.param=[self.actor.param,self.critic.param]
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.actor_old(s)
    
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
        nn.assign_param(self.actor_old.param, self.actor.param)
        return


class PPO_(nn.RL):
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps,alpha,processes,temp=10.0):
        super().__init__()
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.actor_old=actor(state_dim,hidden_dim,action_dim)
        self.controller = Controller()
        nn.assign_param(self.actor_old.param,self.actor.param)
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.alpha=alpha
        self.temp = temp
        self.param=[self.actor.param,self.critic.param,self.controller.param]
        self.env=[gym.make('CartPole-v0') for _ in range(processes)]
    
    def action(self,s):
        return self.actor_old(s)
    
    def window_size(self,p):
        ratio_score = tf.reduce_sum(tf.abs(self.prioritized_replay.ratio_list[p]-1.0))
        td_score = tf.reduce_sum(self.prioritized_replay.TD_list[p])
        scores = self.lambda_ * self.prioritized_replay.TD_list[p] + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio_list[p] - 1.0)
        weights = tf.pow(scores + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([ratio_score, td_score, ess, len(self.prioritized_replay.ratio)], (1,4))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        return self.controller(features)
    
    def window_size_fn(self,p):
        ratio_score = tf.reduce_sum(tf.abs(self.prioritized_replay.ratio_list[p]-1.0))
        td_score = tf.reduce_sum(self.prioritized_replay.TD_list[p])
        scores = self.lambda_ * self.prioritized_replay.TD_list[p] + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio_list[p] - 1.0)
        weights = tf.pow(scores + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([ratio_score, td_score, ess, len(self.prioritized_replay.ratio)], (1,4))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        return self.controller(features)
    
    def batch_size_fn(self):
        if self.batch_counter%777:
            return self.adjust_batch_size()
        return self.adjust_batch_size()
    
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
        self.controller.max_w = len(self.prioritized_replay.ratio)
        ratio_score = tf.reduce_sum(tf.abs(self.prioritized_replay.ratio-1.0))
        td_score = tf.reduce_sum(self.prioritized_replay.TD)
        score = ratio_score + td_score
        scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
        weights = tf.pow(scores + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([ratio_score, td_score, ess, len(self.prioritized_replay.ratio)], (1,4))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        w = self.controller(features)
        idx = tf.cast(tf.range(len(self.prioritized_replay.ratio), w.dtype))
        m = tf.sigmoid((idx - w) / self.temp)
        controller_loss = tf.reduce_mean(m * score)
        self.prioritized_replay.update(TD,ratio)
        return tf.reduce_mean(clip_loss)+tf.reduce_mean((TD)**2)+controller_loss
    
    def update_param(self):
        nn.assign_param(self.actor_old.param, self.actor.param)
        return