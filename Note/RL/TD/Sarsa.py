import numpy as np
import pickle


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
        self.epi_num=0
        self.total_episode=0
        self.total_time=0


    def init(self,dtype=np.int32):
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
    
    
    def td(self,q,episode):
        for s,a,next_s,r in episode:
            action_prob=self.epsilon_greedy_policy(q,next_s,self.action_one)
            next_a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
            q[s][a]=q[s][a]+self.alpha*(r+self.discount*q[next_s][next_a]-q[s][a])
        return q
    
    
    def _explore(self,episode_num,q,s,action,action_one):
        a=0
        episode=[]
        _episode=[]
        for _ in range(episode_num):
            if self.episode_step==None:
                while True:
                    action_prob=self.epsilon_greedy_policy(q,s,action_one)
                    a=np.random.choice(action,p=action_prob)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    temp=q[s][a]
                    self.delta+=np.abs(q[s][a]-temp)
                    _episode.append([s,a,next_s,r])
                    if end:
                        self.delta+=self.delta/a
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r,end])
                        _episode.append([s,a,next_s,r])
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
                    _episode.append([s,a,next_s,r])
                    if end:
                        self.delta+=self.delta/a
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r,end])
                        _episode.append([s,a,next_s,r])
                        break
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
                    s=next_s
                    a+=1
            if self.save_episode==True:
                self.episode.append(episode)
            self.epi_num+=1
        return _episode
    
    
    def explore(self,episode_num):
        s=int(np.random.uniform(0,len(self.state_name)))
        return self._episode(episode_num,self.q,s,self.action,self.action_prob)
    
    
    def update_q(self,q,episode):
        return self.td(q,episode)
    
    
    def learn(self,episode,i):
        self.delta=0
        self.q=self.update_q(self.q,episode)
        self.delta=self.delta/(i+1)
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
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
