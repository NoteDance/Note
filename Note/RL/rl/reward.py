import numpy as np


class reward:
    def __init__(self,agent,env,platform):
        self.agent=agent
        self.env=env
        self.platform=platform
        self.end_flag=False
        
    
    def get_reward(self,max_step=None,seed=None):
        reward=0
        if seed==None:
            s=self.env.reset()
        else:
            s=self.env.reset(seed=seed)
        if max_step!=None:
            for i in range(max_step):
                if self.end_flag==True:
                    break
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
                s=next_s
                reward+=r
                try:
                    if self.nn.stop!=None:
                        if self.nn.stop(next_s):
                            break
                except Exception:
                    pass
                if done:
                    break
            return reward
        else:
            while True:
                if self.end_flag==True:
                    break
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
                s=next_s
                reward+=r
                try:
                    if self.nn.stop!=None:
                        if self.nn.stop(next_s):
                            break
                except Exception:
                    pass
                if done:
                    break
            return reward
