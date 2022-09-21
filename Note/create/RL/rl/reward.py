import torch
import numpy as np


def reward(nn,env,max_step=None,device='cuda'):
    r=0
    state=env.reset()
    if max_step!=None:
        for i in range(max_step):
            state=torch.tensor(np.expand_dims(state,0),dtype=torch.float)
            try:
                if nn.nn!=None:
                    pass
                action_prob=nn.nn(state.to(device))
                action=np.argmax(action_prob).numpy()
            except AttributeError:
                action=nn.actor(state.to(device))
                action=np.squeeze(action).numpy()
            state,reward,done,_=env.step(action)
            state=state
            r+=reward
            if done:
                break
        return r
    else:
        while True:
            state=torch.tensor(np.expand_dims(state,0),dtype=torch.float)
            try:
                if nn.nn!=None:
                    pass
                action_prob=nn.nn(state.to(device))
                action=np.argmax(action_prob).numpy()
            except AttributeError:
                action=nn.actor(state.to(device))
                action=np.squeeze(action).numpy()
            state,reward,done,_=env.step(action)
            state=state
            r+=reward
            if done:
                break
        return r