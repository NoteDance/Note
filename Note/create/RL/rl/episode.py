import torch
import numpy as np


class episode:
    def __init__(self,agent,env):
        self.agent=agent
        self.env=env
        self.end_flag=False


    def get_episode(self,seed=None):
        if seed==None:
            s=self.env.reset()
        else:
            s=self.env.reset(seed=seed)
        episode=[]
        while True:
            try:
                if self.agent.nn!=None:
                    pass
                try:
                    if self.agent.action!=None:
                        pass
                    s=np.expand_dims(s,axis=0)
                    s=torch.tensor(s,dtype=torch.float).to(self.agent.device_d)
                    a=self.agent.action(s).detach().numpy()
                except AttributeError:
                    s=np.expand_dims(s,axis=0)
                    s=torch.tensor(s,dtype=torch.float).to(self.agent.device_d)
                    a=self.agent.nn(s).detach().numpy().argmax()
                next_s,r,done,_=self.env.step(a)
            except AttributeError:
                s=np.expand_dims(s,axis=0)
                a=self.agent.actor(s).detach().numpy()
                a=np.squeeze(a)
                next_s,r,done,_=self.env.step(a)
            try:
                if self.nn.stop!=None:
                    pass
                if self.nn.stop(next_s):
                    break
            except AttributeError:
                pass
            if self.end_flag==True:
                break
            elif done:
                episode.append([s,a,next_s,r])
                episode.append('done')
                break
            else:
                episode.append([s,a,next_s,r])
            s=next_s
        return episode
