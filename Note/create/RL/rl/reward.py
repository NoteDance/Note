import torch
import numpy as np


class reward:
    def __init__(self,nn,env,device='cuda'):
        self.nn=nn
        self.env=env
        self.end_flag=False
        self.device=device
        
    
    def reward(self,max_step=None):
        r=0
        state=self.env.reset()
        if max_step!=None:
            for i in range(max_step):
                if self.end_flag==True:
                    break
                state=torch.tensor(np.expand_dims(state,0),dtype=torch.float).to(self.device)
                try:
                    if self.nn.nn!=None:
                        pass
                    try:
                        if self.nn.action!=None:
                            pass
                        action=self.nn.action(state)
                    except AttributeError:
                        action_prob=self.nn.nn(state)
                        action=np.argmax(action_prob).numpy()
                except AttributeError:
                    action=self.nn.actor(state)
                    action=np.squeeze(action).numpy()
                state,reward,done,_=self.env.step(action)
                state=state
                r+=reward
                if done:
                    break
            return r
        else:
            while True:
                if self.end_flag==True:
                    break
                state=torch.tensor(np.expand_dims(state,0),dtype=torch.float).to(self.device)
                try:
                    if self.nn.nn!=None:
                        pass
                    action_prob=self.nn.nn(state)
                    action=np.argmax(action_prob).numpy()
                except AttributeError:
                    action=self.nn.actor(state)
                    action=np.squeeze(action).numpy()
                state,reward,done,_=self.env.step(action)
                state=state
                r+=reward
                if done:
                    break
            return r
