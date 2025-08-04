import tensorflow as tf
import numpy as np


class pr:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,lambda_,alpha,batch):
        if self.PPO:
            scores=self.lambda_*self.TD+(1.0-self.lambda_)*tf.abs(self.ratio-1.0)
            prios=tf.pow(scores+epsilon,alpha)
            p=prios/tf.reduce_sum(prios)
        else:
            prios=(self.TD+epsilon)**alpha
            p=prios/tf.reduce_sum(prios)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p.numpy(),replace=False)
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update_TD(self,TD,ratio=None):
        if self.PPO:
            if self.pool_network==True:
                for i in range(len(self.index)):
                    self.ratio[7][self.index[i]].assign(ratio[i])
                    self.TD[7][self.index[i]].assign(tf.abs(TD[i]))
            else:
                for i in range(len(self.index)):
                    self.ratio[self.index[i]].assign(ratio[i])
                    self.TD[self.index[i]].assign(tf.abs(TD[i]))
        else:
            if self.pool_network==True:
                for i in range(len(self.index)):
                    self.TD[7][self.index[i]].assign(tf.abs(TD[i]))
            else:
                for i in range(len(self.index)):
                    self.TD[self.index[i]].assign(tf.abs(TD[i]))
        return


class pr_:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.lambda_=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch):
        if self.PPO:
            scores=self.lambda_*self.TD+(1.0-self.lambda_)*tf.abs(self.ratio-1.0)
            prios=tf.pow(scores+epsilon,alpha)
            p=prios/np.sum(prios)
        else:
            prios=(self.TD+epsilon)**alpha
            p=prios/np.sum(prios)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p,replace=False)
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update_TD(self,TD,ratio=None):
        if self.PPO:
            if self.pool_network==True:
                for i in range(len(self.index)):
                    self.ratio[7][self.index[i]].assign(ratio[i])
                    self.TD[7][self.index[i]].assign(tf.abs(TD[i]))
            else:
                for i in range(len(self.index)):
                    self.ratio[self.index[i]].assign(ratio[i])
                    self.TD[self.index[i]].assign(tf.abs(TD[i]))
        else:
            if self.pool_network==True:
                for i in range(len(self.index)):
                    self.TD[7][self.index[i]].assign(tf.abs(TD[i]))
            else:
                for i in range(len(self.index)):
                    self.TD[self.index[i]].assign(tf.abs(TD[i]))
        return


class pr_mp:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.lambda_=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch,p):
        if self.PPO:
            scores=self.lambda_*self.TD[p]+(1.0-self.lambda_)*tf.abs(self.ratio[p]-1.0)
            prios=tf.pow(scores+epsilon,alpha)
            prob=prios/tf.reduce_sum(prios)
        else:
            prios=(self.TD[p]+epsilon)**alpha
            prob=prios/tf.reduce_sum(prios)
        self.index[p]=np.random.choice(np.arange(len(state_pool)),size=[batch],p=prob.numpy(),replace=False)
        return state_pool[self.index[p]],action_pool[self.index[p]],next_state_pool[self.index[p]],reward_pool[self.index[p]],done_pool[self.index[p]]
    
    
    def update_TD(self,TD,ratio=None,p=None):
        if self.PPO:
            for i in range(len(self.index[p])):
                self.ratio[p][self.index[p][i]].assign(ratio[i])
                self.TD[p][self.index[p][i]].assign(tf.abs(TD[i]))
        else:
            for i in range(len(self.index[p])):
                self.TD[p][self.index[p][i]].assign(tf.abs(TD[i]))
        return


class pr_mp_:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.lambda_=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch,p):
        if self.PPO:
            scores=self.lambda_*self.TD[p]+(1.0-self.lambda_)*tf.abs(self.ratio[p]-1.0)
            prios=tf.pow(scores+epsilon,alpha)
            prob=prios/np.sum(prios)
        else:
            prios=(self.TD[p]+epsilon)**alpha
            prob=prios/np.sum(prios)
        self.index[p]=np.random.choice(np.arange(len(state_pool)),size=[batch],p=prob,replace=False)
        return state_pool[self.index[p]],action_pool[self.index[p]],next_state_pool[self.index[p]],reward_pool[self.index[p]],done_pool[self.index[p]]
    
    
    def update_TD(self,TD,ratio=None,p=None):
        if self.PPO:
            for i in range(len(self.index[p])):
                self.ratio[p][self.index[p][i]].assign(ratio[i])
                self.TD[p][self.index[p][i]].assign(np.abs(TD[i]))
        else:
            for i in range(len(self.index[p])):
                self.TD[p][self.index[p][i]].assign(np.abs(TD[i]))
        return
