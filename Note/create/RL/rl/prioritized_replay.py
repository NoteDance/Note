import numpy as np


class pr:
    def __init__(self):
        self.TD=np.array(0)
        self.index=None
        
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch):
        p=(self.TD[1:]+epsilon)**alpha/np.sum(self.TD[1:]+epsilon)**alpha
        self.index=np.random.choice(np.arange(len(state_pool),dtype=np.int8),size=[batch],p=p)+1
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update_TD(self,TD):
        for i in range(len(self.index)):
            self.TD[self.index[i]]=TD[i]
        return


class pr_mt:
    def __init__(self):
        self.TD=[]
        self.index=[]
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch,t):
        p=(self.TD[t][1:]+epsilon)**alpha/np.sum(self.TD[t][1:]+epsilon)**alpha
        self.index[t]=np.random.choice(np.arange(len(state_pool),dtype=np.int8),size=[batch],p=p)+1
        return state_pool[self.index[t]],action_pool[self.index[t]],next_state_pool[self.index[t]],reward_pool[self.index[t]],done_pool[self.index[t]]
    
    
    def update_TD(self,TD,t):
        for i in range(len(self.index[t])):
            self.TD[t][self.index[t][i]]=TD[i]
        return
