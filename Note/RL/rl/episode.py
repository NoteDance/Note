import numpy as np


class episode:
    def __init__(self,agent,env,platform):
        self.agent=agent
        self.env=env
        self.platform=platform
        self.end_flag=False
    
    
    def get_episode(self,seed=None):
        if seed==None:
            s=self.env.reset()
        else:
            s=self.env.reset(seed=seed)
        episode=[]
        while True:
            try:
                try:
                   if self.platform.DType!=None: 
                       s=np.expand_dims(s,axis=0)
                       a=np.argmax(self.agent.nn.fp(s))
                except Exception:
                    s=np.expand_dims(s,axis=0)
                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                    a=self.agent.nn(s).detach().numpy().argmax()
            except Exception as e:
                first_exception=e
                try:
                   if self.agent.nn!=None: 
                       raise first_exception
                except Exception:
                    try:
                        try:
                            if self.agent.action!=None:
                                try:
                                   if self.platform.DType!=None: 
                                       s=np.expand_dims(s,axis=0)
                                       a=self.agent.action(s).numpy()
                                except Exception:
                                    s=np.expand_dims(s,axis=0)
                                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                                    a=self.agent.action(s).detach().numpy()
                        except Exception:
                            try:
                                if self.platform.DType!=None: 
                                    s=np.expand_dims(s,axis=0)
                                    a=self.agent.actor.fp(s).numpy()
                                    a=np.squeeze(a)
                            except Exception:
                                s=np.expand_dims(s,axis=0)
                                s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                                a=self.agent.actor(s).detach().numpy()
                                a=np.squeeze(a)
                    except Exception as e:
                        raise e
            next_s,r,done,_=self.env.step(a)
            try:
                if self.nn.stop!=None:
                    if self.nn.stop(next_s):
                        break
            except Exception:
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
