import numpy as np
import time


class RT_Q_learning:
    def __init__(self,q,state_name,action_name,search_space,epsilon,alpha,discount,dst,episode_step=None,save_episode=True):
        self.q=q
        self.episode=[]
        self.state_name=state_name
        self.action_name=action_name
        self.search_space=search_space
        self.epsilon=epsilon
        self.alpha=alpha
        self.discount=discount
        self.dst=dst
        self.episode_step=episode_step
        self.save_episode=save_episode
    
    
    def init(self):
        self.t3=time.time()
        if len(self.state_name)>self.state_len:
            self.state=np.concatenate(self.state,np.arange(len(self.state_name)-self.state_len,dtype=np.int8)+len(self.state_len))
            self.state_one=np.concatenate(self.state_one,np.ones(len(self.state_name)-self.state_len,dtype=np.int8))
            self.state_prob=self.state_one/len(self.state_name)
            self.action=np.concatenate(self.action,np.arange(len(self.action_name)-self.action_len,dtype=np.int8)+len(self.action_len))
            self.action_prob=np.concatenate(self.action_prob,np.ones(len(self.action_name)-self.action_len,dtype=np.int8))
        else:
            self.state=np.arange(len(self.state_name),dtype=np.int8)
            self.state_one=np.ones(len(self.state_name),dtype=np.int8)
            self.state_prob=self.state_one/len(self.state_name)
            self.action=np.arange(len(self.action_name),dtype=np.int8)
            self.action_prob=np.ones(len(self.action_name),dtype=np.int8)
        if len(self.state_name)>self.q.shape[0] or len(self.action_name)>self.q.shape[1]:
            self.q=np.concatenate([self.q,np.zeros([len(self.state_name),len(self.action_name)-self.action_len],dtype=self.q.dtype)],axis=1)
            self.q=np.concatenate([self.q,np.zeros([len(self.state_name)-self.state_len,len(self.action_name)],dtype=self.q.dtype)])
            self.q=self.q.numpy()
        self.t4=time.time()
        return


    def epsilon_greedy_policy(self,q,s,action_p):
        action_prob=action_p
        action_prob=action_prob*self.epsilon/np.sum(action_p)
        best_a=np.argmax(q[s])
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def td(self,q,s,a,next_s,reward):
        q[s][a]=q[s][a]+self.alpha*(reward+self.discount*np.max(q[next_s])-q[s][a])
        return q,next_s
    
    
    def RT_update_q(self,q,s,action,action_p):
        a=0
        delta=0
        while True:
            t1=time.time()
            a+=1
            action_prob=self.epsilon_greedy_policy(q,s,action_p)
            a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
            next_s,reward,end=self.search_space[self.state_name[s]][self.action_name[a]]
            temp=q[s][a]
            delta+=np.abs(q[s][a]-temp)
            q,next_s=self.td(q,s,a,next_s,reward)
            s=next_s
            self.dst[0]=delta/self.dst[1]
            self.dst[1]+=1
            if self.save_episode==True and a<=self.episode_step:
                self.episode.append([self.state_name[s],self.action_name[a],reward])
            t2=time.time()
            _time=t2-t1+self.t4-self.t3
            self.dst[2]+=_time
        return
    
    
    def learn(self):
        s=np.random.choice(self.state,p=self.state_prob)
        self.q=self.RT_update_q(self.q,s,self.action,self.action_prob)
        return
