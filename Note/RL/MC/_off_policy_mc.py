import numpy as np
import pickle
import time


class off_policy_mc:
    def __init__(self,q,c,state_name,action_name,search_space,epsilon=None,discount=None,theta=None,episode_step=None,save_episode=True):
        self.q=q
        self.c=c
        self.episode=[]
        self.state_name=state_name
        self.action_name=action_name
        self.search_space=search_space
        self.state_len=len(self.state_name)
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
        
        
    def init(self,dtype=np.int32):
        self.t3=time.time()
        if len(self.state_name)>self.state_len:
            self.state=np.concatenate(self.state,np.arange(len(self.state_name)-self.state_len,dtype=dtype)+len(self.state_len))
            self.state_one=np.concatenate(self.state_one,np.ones(len(self.state_name)-self.state_len,dtype=dtype))
            self.state_prob=self.state_one/len(self.state_name)
            self.action=np.concatenate(self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+len(self.action_len))
            self.action_prob=np.concatenate(self.action_prob,np.ones(len(self.action_name)-self.action_len,dtype=dtype))
        else:
            self.state=np.arange(len(self.state_name),dtype=dtype)
            self.state_one=np.ones(len(self.state_name),dtype=dtype)
            self.state_prob=self.state_one/len(self.state_name)
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_prob=np.ones(len(self.action_name),dtype=dtype)
        if len(self.state_name)>self.q.shape[0] or len(self.action_name)>self.q.shape[1]:
            self.q=np.concatenate([self.q,np.zeros([len(self.state_name),len(self.action_name)-self.action_len],dtype=self.q.dtype)],axis=1)
            self.q=np.concatenate([self.q,np.zeros([len(self.state_name)-self.state_len,len(self.action_name)],dtype=self.q.dtype)])
            self.q=self.q.numpy()
        if len(self.state_name)>self.c.shape[0] or len(self.action_name)>self.c.shape[1]:
            self.c=np.concatenate([self.c,np.zeros([len(self.state_name),len(self.action_name)-self.action_len],dtype=self.c.dtype)],axis=1)
            self.c=np.concatenate([self.c,np.zeros([len(self.state_name)-self.state_len,len(self.action_name)],dtype=self.c.dtype)])
            self.c=self.c.numpy()
        self.t4=time.time()
        return


    def epsilon_greedy_policy(self,q,s,action_p):
        action_prob=action_p
        action_prob=action_prob*self.epsilon/np.sum(action_p)
        best_a=np.argmax(q[s])
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def episode(self,q,s,action,action_p,search_space):
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
        return episode
    
    
    def importance_sampling(self,episode,q,discount,action_p):
        w=1
        temp=0
        a=0
        delta=0
        self.delta=0
        for i,[s,a,r] in enumerate(episode):
            a+=1
            first_visit_index=i
            G=sum(np.power(discount,i)*x[2] for i,x in enumerate(episode[first_visit_index:]))
            self.c[s][a]+=w
            delta+=np.abs(temp-(w/self.c[s][a])*(G-q[s][a]))
            q[s][a]+=(w/self.c[s][a])*(G-q[s][a])
            if a!=np.argmax(q[s]):
                break
            action_prob=self.epsilon_greedy_policy(q,s,action_p)
            w=w*1/action_prob
            temp=(w/self.c[s][a])*(G-q[s][a])
        self.delta+=delta/a
        return q
    
    
    def learn(self,episode_num,path=None,one=True):
        self.delta=0
        for i in range(episode_num):
            t1=time.time()
            s=np.random.choice(self.state,p=self.state_prob)
            e=self.episode(self.q,s,self.action,self.action_prob,self.search_space,self.episode_step)
            self.q=self.importance_sampling(e,self.q,self.discount)
            self.delta=self.delta/(i+1)
            if episode_num%10!=0:
                temp=episode_num-episode_num%10
                temp=int(temp/10)
            else:
                temp=episode_num/10
            if temp==0:
                temp=1
            if i%temp==0:
                print('episode num:{0}   delta:{1:.6f}'.format(i+1,self.delta))
                if path!=None and i%episode_num*2==0:
                    self.save(path,i,one)
            self.episode_num+=1
            self.total_episode+=1
            t2=time.time()
            self.time+=(t2-t1)
            if self.theta!=None and self.delta<=self.theta:
                break
        if self.time<0.5:
            self.time=int(self.time+(self.t4-self.t3))
        else:
            self.time=int(self.time+(self.t4-self.t3))+1
        self.total_time+=self.time
        print()
        print('last delta:{0:.6f}'.format(self.delta))
        print('time:{0}s'.format(self.time))
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.state_len,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.state,output_file)
        pickle.dump(self.state_prob,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_prob,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.theta,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.delta,output_file)
        pickle.dump(self.state_one,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.state_len=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        if self.state_len==len(self.state_name):
            self.state=pickle.load(input_file)
            self.state_prob=pickle.load(input_file)
        if self.action_len==len(self.action_name):
            self.action=pickle.load(input_file)
            self.action_prob=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.state_one=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
