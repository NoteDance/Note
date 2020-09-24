import tensorflow as tf
import numpy as np
import pickle
import time


class Q_learning:
    def __init__(self,q,state_name,action_name,search_space,epsilon=None,alpha=None,discount=None,theta=None,episode_step=None,save_episode=True):
        self.q=q
        self.episode=[]
        self.state_name=state_name
        self.action_name=action_name
        self.search_space=search_space
        self.epsilon=epsilon
        self.alpha=alpha
        self.discount=discount
        self.theta=theta
        self.episode_step=episode_step
        self.save_episode=save_episode
        self.delta=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0


    def epsilon_greedy_policy(self,q,s,action_p):
        action_prob=action_p
        action_prob=action_prob*self.epsilon/np.sum(action_p)
        best_a=np.argmax(q[s])
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def td(self,q,s,a,next_s,r):
        q[s][a]=q[s][a]+self.alpha*(r+self.discount*np.max(q[next_s])-q[s][a])
        return q
    
    
    def update_q(self,q,s,action,action_p):
        a=0
        episode=[]
        if self.episode_step==None:
            while True:
                action_prob=self.epsilon_greedy_policy(q,s,action_p)
                a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                temp=q[s][a]
                self.delta+=np.abs(q[s][a]-temp)
                if end:
                    self.delta+=self.delta/a
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],r])
                q=self.td(q,s,a,next_s,r)
                s=next_s
                a+=1
        else:
            for _ in range(self.episode_step):
                action_prob=self.epsilon_greedy_policy(q,s,action_p)
                a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                temp=q[s][a]
                self.delta+=np.abs(q[s][a]-temp)
                if end:
                    self.delta+=self.delta/a
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],r])
                q=self.td(q,s,a,next_s,r)
                s=next_s
                a+=1
        if self.save_episode==True:
            self.episode.append(episode)
        return q
    
    
    def learn(self,episode_num,path=None,one=True):
        self.delta=0
        state=np.arange(len(self.state_name),dtype=np.int8)
        state_prob=np.ones(len(self.state_name),dtype=np.int8)/len(self.state_name)
        action=np.arange(len(self.action_name),dtype=np.int8)
        action_prob=np.ones(len(self.action_name),dtype=np.int8)
        if len(self.state_name)>self.q.shape[0] or len(self.action_name)>self.q.shape[1]:
            q=self.q*tf.ones([len(self.state_name),len(self.action_name)],dtype=self.q.dtype)[:self.q.shape[0],:self.q.shape[1]]
            self.q=q.numpy()
        for i in range(episode_num):
            t1=time.time()
            s=np.random.choice(state,p=state_prob)
            self.q=self.update_q(self.q,s,action,action_prob)
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
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
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
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.alpha,output_file)
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
        self.epsilon=pickle.load(input_file)
        self.alpha=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
