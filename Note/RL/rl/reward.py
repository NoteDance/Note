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
                if hasattr(self.agent,'nn'):
                    if hasattr(self.platform,'DType'):
                        s=np.expand_dims(s,axis=0)
                        a=np.argmax(self.agent.nn.fp(s))
                    else:
                        s=np.expand_dims(s,axis=0)
                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                        a=self.agent.nn(s).detach().numpy().argmax()
                else:
                    if hasattr(self.agent,'action'):
                        if hasattr(self.platform,'DType'):
                            s=np.expand_dims(s,axis=0)
                            a=self.agent.action(s).numpy()
                        else:
                            s=np.expand_dims(s,axis=0)
                            a=self.agent.action(s).detach().numpy()
                    else:
                        if hasattr(self.platform,'DType'):
                            s=np.expand_dims(s,axis=0)
                            a=self.agent.actor.fp(s).numpy()
                            a=np.squeeze(a)
                        else:
                            s=np.expand_dims(s,axis=0)
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                            a=self.agent.actor(s).detach().numpy()
                            a=np.squeeze(a)
                next_s,r,done,_=self.env.step(a)
                s=next_s
                reward+=r
                if hasattr(self.nn,'stop'):
                    if self.nn.stop(next_s):
                        break
                if done:
                    break
            return reward
        else:
            while True:
                if self.end_flag==True:
                    break
                if hasattr(self.agent,'nn'):
                    if hasattr(self.platform,'DType'):
                        s=np.expand_dims(s,axis=0)
                        a=np.argmax(self.agent.nn.fp(s))
                    else:
                        s=np.expand_dims(s,axis=0)
                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                        a=self.agent.nn(s).detach().numpy().argmax()
                else:
                    if hasattr(self.agent,'action'):
                        if hasattr(self.platform,'DType'):
                            s=np.expand_dims(s,axis=0)
                            a=self.agent.action(s).numpy()
                        else:
                            s=np.expand_dims(s,axis=0)
                            a=self.agent.action(s).detach().numpy()
                    else:
                        if hasattr(self.platform,'DType'):
                            s=np.expand_dims(s,axis=0)
                            a=self.agent.actor.fp(s).numpy()
                            a=np.squeeze(a)
                        else:
                            s=np.expand_dims(s,axis=0)
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.agent.device)
                            a=self.agent.actor(s).detach().numpy()
                            a=np.squeeze(a)
                next_s,r,done,_=self.env.step(a)
                s=next_s
                reward+=r
                if hasattr(self.nn,'stop'):
                    if self.nn.stop(next_s):
                        break
                if done:
                    break
            return reward
