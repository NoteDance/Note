import tensorflow as tf
import numpy as np


class pr:
    def __init__(self):
        self.TD=tf.Variable(7.)
        self.index=None
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch):
        p=(self.TD+epsilon)**alpha/tf.reduce_sum((self.TD+epsilon)**alpha)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p.numpy())
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update_TD(self,TD):
        if self.pool_network==True:
            for i in range(len(self.index)):
                self.TD[7][self.index[i]].assign(TD[i])
        else:
            for i in range(len(self.index)):
                self.TD[self.index[i]].assign(TD[i])
        return


class pr_:
    def __init__(self):
        self.TD=7.
        self.index=None
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch):
        p=(self.TD+epsilon)**alpha/np.sum((self.TD+epsilon)**alpha)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p)
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update_TD(self,TD):
        if self.pool_network==True:
            for i in range(len(self.index)):
                self.TD[7][self.index[i]]=TD[i]
        else:
            for i in range(len(self.index)):
                self.TD[self.index[i]]=TD[i]
        return


class pr_mp:
    def __init__(self):
        self.TD=None
        self.index=None
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch,p):
        prob=(self.TD[p]+epsilon)**alpha/tf.reduce_sum((self.TD[p]+epsilon)**alpha)
        self.index[p]=np.random.choice(np.arange(len(state_pool)),size=[batch],p=prob.numpy())
        return state_pool[self.index[p]],action_pool[self.index[p]],next_state_pool[self.index[p]],reward_pool[self.index[p]],done_pool[self.index[p]]
    
    
    def update_TD(self,TD,p):
        for i in range(len(self.index[p])):
            self.TD[p][self.index[p][i]].assign(TD[i])
        return
