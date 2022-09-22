import numpy as np


class reward:
    def __init__(self,agent,env):
        self.agent=agent
        self.env=env
        self.end_flag=False
        
    
    def reward(self,max_step=None):
        r=0
        state=self.env.reset()
        if max_step!=None:
            for i in range(max_step):
                if self.end_flag==True:
                    break
                state=np.expand_dims(state,0)
                try:
                    if self.agent.nn!=None:
                        pass
                    try:
                        if self.agent.action!=None:
                            pass
                        action=self.agent.action(state)
                    except AttributeError:
                        action_prob=self.agent.nn(state)
                        action=np.argmax(action_prob).numpy()
                except AttributeError:
                    action=self.agent.actor(state)
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
                state=np.expand_dims(state,0)
                try:
                    if self.agent.nn!=None:
                        pass
                    action_prob=self.agent.nn(state)
                    action=np.argmax(action_prob).numpy()
                except AttributeError:
                    action=self.agent.actor(state)
                    action=np.squeeze(action).numpy()
                state,reward,done,_=self.env.step(action)
                state=state
                r+=reward
                if done:
                    break
            return r
