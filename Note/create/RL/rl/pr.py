import numpy as np


def pr(state_pool,action_pool,next_state_pool,reward_pool,t,pool_size,batch,K,alpha,beta,p=None):
    if p==None:
        p=np.ones([len(state_pool)],dtype=np.float32)
        prob=p**alpha/np.sum(p**alpha)
        w=(pool_size*prob)**-beta
        w=w/np.max(w)
    else:
        p=np.concatenate([p,np.ones([len(state_pool)-len(p)])*np.max(p)])
        prob=p**alpha/np.sum(p**alpha)
        w=(pool_size*prob)**-beta
        w=w/np.max(w)
    if t%K==0:
        index=np.random.choice(np.arange(len(state_pool),dtype=np.int8),size=[pool_size],p=prob)
    t+=1
    return state_pool[index],action_pool[index],next_state_pool[index],reward_pool[index]