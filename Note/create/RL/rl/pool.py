import numpy as np


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