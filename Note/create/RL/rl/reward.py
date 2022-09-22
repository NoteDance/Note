import numpy as np


class reward:
    def __init__(self,agent,env):
        self.agent=agent
        self.env=env
        self.end_flag=False
        
    
    def reward(self,max_step=None):
        reward=0
        s=self.env.reset()
        if max_step!=None:
            for i in range(max_step):
                if self.end_flag==True:
                    break
                s=np.expand_dims(s,0)
                try:
                    if self.agent.nn!=None:
                        pass
                    try:
                        if self.agent.action!=None:
                            pass
                        a=self.agent.action(s)
                    except AttributeError:
                        action_prob=self.agent.nn(s)
                        a=np.argmax(action_prob).numpy()
                except AttributeError:
                    a=self.agent.actor(s)
                    a=np.squeeze(a).numpy()
                next_s,r,done,_=self.env.step(a)
                s=next_s
                reward+=r
                try:
                    if self.nn.stop!=None:
                        pass
                    if self.nn.stop(next_s):
                        break
                except AttributeError:
                    pass
                if done:
                    break
            return r
        else:
            while True:
                if self.end_flag==True:
                    break
                s=np.expand_dims(s,0)
                try:
                    if self.agent.nn!=None:
                        pass
                    action_prob=self.agent.nn(s)
                    a=np.argmax(action_prob).numpy()
                except AttributeError:
                    a=self.agent.actor(s)
                    a=np.squeeze(a).numpy()
                next_s,r,done,_=self.env.step(a)
                s=next_s
                reward+=r
                try:
                    if self.nn.stop!=None:
                        pass
                    if self.nn.stop(next_s):
                        break
                except AttributeError:
                    pass
                if done:
                    break
            return r
