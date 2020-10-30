import tensorflow as tf
import numpy as np
import time


class RT_Sarsa:
    def __init__(self,q,state_name,action_name,search_space,epsilon,alpha,discount,dst,episode_step=None,save_episode=True):
        self.q=q
        if len(state_name)>q.shape[0] or len(action_name)>q.shape[1]:
            q=q*tf.ones([len(state_name),len(action_name)],dtype=q.dtype)[:q.shape[0],:q.shape[1]]
            self.q=q.numpy()
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
        self.state=np.arange(len(self.state_name),dtype=np.int8)
        self.state_prob=np.ones(len(self.state_name),dtype=np.int8)/len(self.state_name)
        self.action=np.arange(len(self.action_name),dtype=np.int8)
        self.action_prob=np.ones(len(self.action_name),dtype=np.int8)
        return


    def epsilon_greedy_policy(self,q,s,action_p):
        action_prob=action_p
        action_prob=action_prob*self.epsilon/np.sum(action_p)
        best_a=np.argmax(q[s])
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def td(self,q,s,a,next_s,reward,action_p):
        action_prob=self.epsilon_greedy_policy(q,next_s,action_p)
        next_a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
        q[s][a]=q[s][a]+self.alpha*(reward+self.discount*q[next_s][next_a]-q[s][a])
        return q,next_s
    
    
    def RT_update_q(self,q,s,action,action_p):
        a=0
        delta=0
        while True:
            t1=time.time()
            a+=1
            action_prob=self.epsilon_greedy_policy(q,s,action_p)
            a=np.random.choice(action,p=action_prob)
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
            _time=t2-t1
            self.dst[2]+=_time
        return
    
    
    def learn(self):
        s=np.random.choice(self.state,p=self.state_prob)
        self.q=self.RT_update_q(self.q,s,self.action,self.action_prob)
        return
