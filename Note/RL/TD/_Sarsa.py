import numpy as np
import pickle
import time


class Sarsa:
    def __init__(self,q,state_name,action_name,exploration_space,epsilon=None,alpha=None,discount=None,theta=None,episode_step=None,save_episode=True):
        self.q=q
        self.episode=[]
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.action_len=len(self.action_name)
        self.epsilon=epsilon
        self.alpha=alpha
        self.discount=discount
        self.theta=theta
        self.episode_step=episode_step
        self.save_episode=save_episode
        self.delta=0
        self.epi_num=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
        
        
    def init(self,dtype=np.int32):
        self.t3=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            self.action_prob=np.concatenate((self.action_prob,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_prob=np.ones(len(self.action_name),dtype=dtype)
        if len(self.state_name)>self.q.shape[0] or len(self.action_name)>self.q.shape[1]:
            self.q=np.concatenate((self.q,np.zeros([len(self.state_name),len(self.action_name)-self.action_len],dtype=self.q.dtype)),axis=1)
            self.q=np.concatenate((self.q,np.zeros([len(self.state_name)-self.state_len,len(self.action_name)],dtype=self.q.dtype)))
            self.q=self.q.numpy()
        self.t4=time.time()
        return
    
    
    def set_up(self,epsilon=None,alpha=None,discount=None,theta=None,episode_step=None,init=True):
        if epsilon!=None:
            self.epsilon=epsilon
        if alpha!=None:
            self.alpha=alpha
        if discount!=None:
            self.discount=discount
        if theta!=None:
            self.theta=theta
        if episode_step!=None:
            self.episode_step=episode_step
        if init==True:
            self.episode=[]
            self.delta=0
            self.epi_num=0
            self.episode_num=0
            self.total_episode=0
            self.time=0
            self.total_time=0
        return
    

    def epsilon_greedy_policy(self,q,s,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_a=np.argmax(q[s])
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def td(self,q,s,a,next_s,r,action_one):
        action_prob=self.epsilon_greedy_policy(q,next_s,action_one)
        next_a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
        q[s][a]=q[s][a]+self.alpha*(r+self.discount*q[next_s][next_a]-q[s][a])
        return q
    
    
    def update_q(self,q,s,action,action_one):
        a=0
        episode=[]
        if self.episode_step==None:
            while True:
                action_prob=self.epsilon_greedy_policy(q,s,action_one)
                a=np.random.choice(action,p=action_prob)
                next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                temp=q[s][a]
                self.delta+=np.abs(q[s][a]-temp)
                q=self.td(q,s,a,next_s,r,action_one)
                if end:
                    self.delta+=self.delta/a
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
                s=next_s
                a+=1
        else:
            for _ in range(self.episode_step):
                action_prob=self.epsilon_greedy_policy(q,s,action_one)
                a=np.random.choice(action,p=action_prob)
                next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                temp=q[s][a]
                self.delta+=np.abs(q[s][a]-temp)
                q=self.td(q,s,a,next_s,r,action_one)
                if end:
                    self.delta+=self.delta/a
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
                s=next_s
                a+=1
        if self.save_episode==True:
            self.episode.append(episode)
        return q
    
    
    def learn(self,episode_num,path=None,one=True):
        self.delta=0
        for i in range(episode_num):
            t1=time.time()
            s=int(np.random.uniform(0,len(self.state_name)))
            self.q=self.update_q(self.q,s,self.action,self.action_prob)
            self.delta=self.delta/(i+1)
            if episode_num%10!=0:
                d=episode_num-episode_num%10
                d=int(d/10)
            else:
                d=episode_num/10
            if d==0:
                d=1
            if i%d==0:
                print('episode num:{0}   delta:{1:.6f}'.format(i+1,self.delta))
                if path!=None and i%episode_num*2==0:
                    self.save(path,i,one)
            self.epi_num+=1
            self.total_episode+=1
            t2=time.time()
            self.time+=(t2-t1)
            if self.theta!=None and self.delta<=self.theta:
                break
        if path!=None:
            self.save(path)
        if self.time<0.5:
            self.time=int(self.time+(self.t4-self.t3))
        else:
            self.time=int(self.time+(self.t4-self.t3))+1
        self.total_time+=self.time
        print()
        print('last delta:{0:.6f}'.format(self.delta))
        print('time:{0}s'.format(self.time))
        return
    
    
    def save_policy(self,path):
        policy_file=open(path+'.dat','wb')
        pickle.dump(self.q,policy_file)
        policy_file.close()
        return
    
    
    def save_e(self,path):
        episode_file=open(path+'.dat','wb')
        pickle.dump(self.episode,episode_file)
        episode_file.close()
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'\save.dat','wb')
            path=path+'\save.dat'
            index=path.rfind('\\')
            if self.save_episode==True:
                episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
        else:
            output_file=open(path+'\save-{0}.dat'.format(i+1),'wb')
            path=path+'\save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            if self.save_episode==True:
                episode_file=open(path.replace(path[index+1:],'episode-{0}.dat'.format(i+1)),'wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
        self.episode_num=self.epi_num
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_prob,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.alpha,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.theta,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.delta,output_file)
        pickle.dump(self.state_one,output_file)
        pickle.dump(self.episode_num,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,e_path=None):
        input_file=open(s_path,'rb')
        if self.save_episode==True:
            episode_file=open(e_path,'rb')
            self.episode=pickle.load(episode_file)
            episode_file.close()
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_prob=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.alpha=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.state_one=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
