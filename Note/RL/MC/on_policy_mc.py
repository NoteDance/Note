import numpy as np
import pickle
import time


class on_policy_mc:
    def __init__(self,q,state_name,action_name,search_space,epsilon=None,discount=None,theta=None,episode_step=None,save_episode=True):
        self.q=q
        self.episode=[]
        self.r_sum=dict()
        self.r_count=dict()
        self.state_name=state_name
        self.action_name=action_name
        self.search_space=search_space
        self.action_len=len(self.action_name)
        self.epsilon=epsilon
        self.discount=discount
        self.theta=theta
        self.episode_step=episode_step
        self.save_episode=save_episode
        self.delta=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
        
        
    def init(self,dtype):
        t3=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate(self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len)
            self.action_prob=np.concatenate(self.action_prob,np.ones(len(self.action_name)-self.action_len,dtype=dtype))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_prob=np.ones(len(self.action_name),dtype=dtype)
        if len(self.state_name)>self.q.shape[0] or len(self.action_name)>self.q.shape[1]:
            self.q=np.concatenate([self.q,np.zeros([len(self.state_name),len(self.action_name)-self.action_len],dtype=self.q.dtype)],axis=1)
            self.q=np.concatenate([self.q,np.zeros([len(self.state_name)-self.state_len,len(self.action_name)],dtype=self.q.dtype)])
            self.q=self.q.numpy()
        t4=time.time()
        self.time+=t4-t3
        return


    def epsilon_greedy_policy(self,q,s,action_p):
        action_prob=action_p
        action_prob=action_prob*self.epsilon/np.sum(action_p)
        best_a=np.argmax(q[s])
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def _episode(self,q,s,action,action_p,search_space):
        episode=[]
        _episode=[]
        if self.episode_step==None:
            while True:
                action_prob=self.epsilon_greedy_policy(q,s,action_p)
                a=np.random.choice(action,p=action_prob)
                next_s,r,end=search_space[self.state_name[s]][self.action_name[a]]
                episode.append([s,a,r])
                if end:
                    if self.save_episode==True:
                        _episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    _episode.append([self.state_name[s],self.action_name[a],r])
                s=next_s
        else:
            for _ in range(self.episode_step):
                action_prob=self.epsilon_greedy_policy(q,s,action_p)
                a=np.random.choice(action,p=action_prob)
                next_s,r,end=search_space[self.state_name[s]][self.action_name[a]]
                episode.append([s,a,r])
                if end:
                    if self.save_episode==True:
                        _episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    _episode.append([self.state_name[s],self.action_name[a],r])
                s=next_s
        if self.save_episode==True:
            self.episode.append(_episode)
        self.epi_num+=1
        return episode
    
    
    def first_visit(self,episode,q,r_sum,r_count,discount):
        state_action_set=set()
        delta=0
        self.delta=0
        for i,[s,a,r] in enumerate(episode):
            state_action=(s,a)
            first_visit_index=i
            G=sum(np.power(discount,i)*x[2] for i,x in enumerate(episode[first_visit_index:]))
            if state_action not in state_action_set:
                state_action_set.add(state_action)
                if i==0:
                    r_sum[state_action]=G
                    r_count[state_action]=1
                else:
                    r_sum[state_action]+=G
                    r_count[state_action]+=1
                    delta+=np.abs(q[s][a]-r_sum[state_action]/r_count[state_action])
            q[s][a]=r_sum[state_action]/r_count[state_action]
        self.delta+=delta/len(episode)
        return q,r_sum,r_count
    
    
    def episode(self):
        s=int(np.random.uniform(0,len(self.state_name)))
        return self._episode(self.q,s,self.action,self.action_prob,self.search_space,self.episode_step)
    
    
    def learn(self,episode,i):
        self.delta=0
        self.q,self.r_sum,self.r_count=self.first_visit(episode,self.q,self.r_sum,self.r_count,self.discount)
        self.delta=self.delta/(i+1)
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.r_sum,output_file)
        pickle.dump(self.r_count,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_prob,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.theta,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.delta,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.r_sum=pickle.load(input_file)
        self.r_count=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        if self.action_len==len(self.action_name):
            self.action=pickle.load(input_file)
            self.action_prob=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
