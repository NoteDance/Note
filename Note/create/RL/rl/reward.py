import numpy as np


class reward:
    def __init__(self,agent,env):
        self.agent=agent
        self.env=env
        self.end_flag=False
        
    
    def reward(self,max_step=None,seed=None):
        reward=0
        if seed==None:
            s=self.env.reset()
        else:
            s=self.env.reset(seed=seed)
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
                        a=self.agent.action(s).numpy()
                    except AttributeError:
                        action_prob=self.agent.nn.fp(s).numpy()
                        a=np.argmax(action_prob)
                except AttributeError:
                    a=self.agent.actor.fp(s).numpy()
                    a=np.squeeze(a)
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
            return reward
        else:
            while True:
                if self.end_flag==True:
                    break
                s=np.expand_dims(s,0)
                try:
                    if self.agent.nn!=None:
                        pass
                    action_prob=self.agent.nn.fp(s).numpy()
                    a=np.argmax(action_prob)
                except AttributeError:
                    a=self.agent.actor.fp(s).numpy()
                    a=np.squeeze(a)
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
            return reward
