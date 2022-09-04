import numpy as np


def epsilon_greedy_policy(action_num,value,action_one,epsilon):
    action_prob=action_one
    action_prob=action_prob*epsilon/len(action_one)
    best_a=np.argmax(value)
    action_prob[best_a]+=1-epsilon
    return np.random.choice(action_num,p=action_prob)


def pool(state,action,next_state,reward,state_pool=None,action_pool=None,next_state_pool=None,reward_pool=None,pool_size=None):
   if state_pool==None:
       state_pool=np.expand_dims(state,axis=0)
       action_pool=np.expand_dims(action,axis=0)
       next_state_pool=np.expand_dims(next_state,axis=0)
       reward_pool=np.expand_dims(reward,axis=0)
   else:
       state_pool=np.concatenate((state_pool,np.expand_dims(state,axis=0)),0)
       action_pool=np.concatenate((action_pool,np.expand_dims(action,axis=0)),0)
       next_state_pool=np.concatenate((next_state_pool,np.expand_dims(next_state,axis=0)),0)
       reward_pool=np.concatenate((reward_pool,np.expand_dims(reward,axis=0)),0)
   if len(state_pool)>pool_size:
       state_pool=state_pool[1:]
       action_pool=action_pool[1:]
       next_state_pool=next_state_pool[1:]
       reward_pool=reward_pool[1:]
   return state_pool,action_pool,next_state_pool,reward_pool


def update_param(target_net,net,tau=None):
    if tau==None:
        target_net.load_state_dict(net.state_dict())
    else:
        for target_param,param in zip(target_net[0].parameters(),net[0].parameters()):
            target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
        for target_param,param in zip(target_net[1].parameters(),net[1].parameters()):
            target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
    return


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
