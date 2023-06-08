import numpy as np
import traceback


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
                if self.agent.nn!=None:
                    try:
                       if self.platform.DType!=None: 
                           s=np.expand_dims(s,axis=0)
                           a=np.argmax(self.agent.nn.fp(s))
                    except Exception:
                        print(traceback.format_exc())
                        s=np.expand_dims(s,axis=0)
                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                        a=self.agent.nn(s).detach().numpy().argmax()
            except Exception:
                print(traceback.format_exc())
                try:
                    if self.agent.action!=None:
                        try:
                           if self.platform.DType!=None: 
                               s=np.expand_dims(s,axis=0)
                               a=self.agent.action(s).numpy()
                        except Exception:
                            print(traceback.format_exc())
                            s=np.expand_dims(s,axis=0)
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                            a=self.agent.action(s).detach().numpy()
                except Exception:
                    print(traceback.format_exc())
                    try:
                        if self.platform.DType!=None: 
                            s=np.expand_dims(s,axis=0)
                            a=self.agent.actor.fp(s).numpy()
                            a=np.squeeze(a)
                    except Exception:
                        print(traceback.format_exc())
                        s=np.expand_dims(s,axis=0)
                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                        a=self.agent.actor(s).detach().numpy()
                        a=np.squeeze(a)
            next_s,r,done,_=self.env.step(a)
            try:
                if self.nn.stop!=None:
                    if self.nn.stop(next_s):
                        break
            except Exception:
                print(traceback.format_exc())
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
