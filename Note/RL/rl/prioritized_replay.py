import tensorflow as tf
import numpy as np


class pr:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,lambda_,alpha,batch):
        if self.PPO:
            scores=self.lambda_*self.TD+(1.0-self.lambda_)*np.abs(self.ratio-1.0)
            prios=np.pow(scores+1e-7,alpha)
            p=prios/np.sum(prios)
        else:
            prios=(self.TD+1e-7)**alpha
            p=prios/np.sum(prios)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p,replace=False)
        try:
            self.batch.assign(batch)
        except Exception:
            self.batch=batch
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update(self,TD=None,ratio=None):
        if self.PPO:
            if TD is not None:
                TD=tf.cast(TD,tf.float32)
                ratio=tf.cast(ratio,tf.float32)
                self.TD_[:self.batch].assign(TD)
                self.ratio_[:self.batch].assign(ratio)
            else:
                self.ratio[self.index]=self.ratio_[:self.batch]
                self.TD[self.index]=np.abs(self.TD_[:self.batch])
        else:
            if TD is not None:
                TD=tf.cast(TD,tf.float32)
                self.TD_[:self.batch].assign(TD)
            else:
                self.TD[self.index]=np.abs(self.TD_[:self.batch])
        return


class pr_mp:
    def __init__(self):
        self.TD=None
        self.lambda_=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,alpha,batch,p):
        prios=(self.TD[p]+1e-7)**alpha
        prob=prios/np.sum(prios)
        self.index[p]=np.random.choice(np.arange(len(state_pool)),size=[batch],p=prob,replace=False)
        self.batch=batch
        return state_pool[self.index[p]],action_pool[self.index[p]],next_state_pool[self.index[p]],reward_pool[self.index[p]],done_pool[self.index[p]]
    
    
    def update(self,TD=None,ratio=None,p=None):
        if TD is not None:
            try:
                TD=tf.cast(TD,tf.float32)
                self.TD_[:self.batch].assign(TD)
            except Exception:
                self.TD_[:self.batch]=TD
        else:
            self.TD[p][self.index[p]]=np.abs(self.TD_)
        return
