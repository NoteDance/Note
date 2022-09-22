import numpy as np


class episode:
    def __init__(self,agent,env):
        self.agent=agent
        self.env=env
        self.end_flag=False
    
    
    def get_episode(self):
        s=self.env.reset()
        episode=[]
        while True:
            try:
                if self.agent.nn!=None:
                    pass
                try:
                    if self.agent.action!=None:
                        pass
                    s=np.expand_dims(s,axis=0)
                    a=self.agent.action(s)
                except AttributeError:
                    s=np.expand_dims(s,axis=0)
                    a=np.argmax(self.agent.nn(s)).numpy()
                next_s,r,done=self.env(a)
            except AttributeError:
                s=np.expand_dims(s,axis=0)
                a=self.agent.actor(s).numpy()
                a=np.squeeze(a)
                next_s,r,done=self.agent.env(a)
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
