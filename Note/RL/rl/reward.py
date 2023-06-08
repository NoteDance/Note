import numpy as np
import traceback


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
                s=next_s
                reward+=r
                try:
                    if self.nn.stop!=None:
                        if self.nn.stop(next_s):
                            break
                except Exception:
                    print(traceback.format_exc())
                    pass
                if done:
                    break
            return reward
        else:
            while True:
                if self.end_flag==True:
                    break
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
                s=next_s
                reward+=r
                try:
                    if self.nn.stop!=None:
                        if self.nn.stop(next_s):
                            break
                except Exception:
                    print(traceback.format_exc())
                    pass
                if done:
                    break
            return reward
