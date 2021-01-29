import tensorflow as tf
import numpy as np
import pickle
import time


class A3C_Q_learning:
    def __init__(self,value_net,value_p,target_p,state,state_name,action_name,exploration_space,Tmax=None,It=None,Ia=None,alpha=None,lr=None,epsilon=None,discount=None,episode_step=None,save_episode=True):
        self.value_net=value_net
        self.value_p=value_p
        self.target_p=target_p
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.action_len=len(self.action_name)
        self.discount=discount
        self.episode_step=episode_step
        self.lr=lr
        self.T=0
        self.Tmax=Tmax
        self.It=It
        self.Ia=Ia
        self.alpha=alpha
        self.save_episode=save_episode
        self.epsilon=epsilon
        self.total_episode=0
    
    
    def init(self,dtype=np.int32):
        self.t1=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_one=np.ones(len(self.action_name),dtype=dtype)
        self.t2=time.time()
        return
    
    
    def set_up(self,value_p=None,target_p=None,Tmax=None,It=None,Ia=None,alpha=None,lr=None,init=True):
        if value_p!=None:
            self.value_p=value_p
            self.target_p=target_p
        if Tmax!=None:
            self.Tmax=Tmax
        if It!=None:
            self.It=It
        if Ia!=None:
            self.Ia=Ia
        if alpha!=None:
            self.alpha=alpha
        if lr!=None:
            self.lr=lr
        if init==True:
            self.episode=[]
        return
    
    
    def epsilon_greedy_policy(self,s,action_one,epsilon):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_a=np.argmax(self.value_net(self.state[self.state_name[s]]))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def update_parameter(self):
        for i in range(len(self.value_p)):
            self.target_p[i]=self.value_p[i].copy()
        return
    
    
    def _loss(self,s,a,next_s,r):
        return ((r+self.discount*tf.reduce_max(self.value_net(next_s,self.target_p)))-self.value_net(s,self.value_p)[a])**2
    
    
    def learn(self,episode_num,epsilon):
        t=0
        for i in range(episode_num):
            g=[0 for _ in range(len(self.value_p))]
            gradient=[0 for _ in range(len(self.value_p))]
            episode=[]
            s=int(np.random.uniform(0,len(self.state_name)))
            if self.episode_step==None:
                while True:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one,epsilon)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                    elif self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    loss=self._loss(self.state[self.state_name[s]],a,self.state[self.state_name[next_s]],r)
                    self.loss=loss
                    with tf.GradientTape() as tape:
                        _gradient=tape.gradient(loss,self.value_p)
                        for i in range(len(_gradient)):
                            g[i]=g[i]+(self.alpha*g[i]+(1-self.alpha)*_gradient[i]**2)
                            gradient[i]=gradient[i]+self.lr*_gradient[i]/tf.math.sqrt(g[i]+self.epsilon)
                    s=next_s
                    self.T+=1
                    t+=1
                    if self.T>self.Tmax:
                        return
                    if self.T%self.It==0:
                        self.update_parameter()
                    elif t%self.Ia==0 or end:
                        for i in range(len(self.value_p)):
                            self.value_p[i]=self.value_p[i]+gradient[i]
                            gradient=[0 for _ in range(len(self.value_p))]
                            g=[0 for _ in range(len(self.value_p))]
            else:
                for _ in range(self.episode_step):
                    action_prob=self.epsilon_greedy_policy(s,self.action_one,epsilon)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                    elif self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    loss=self._loss(self.state[self.state_name[s]],a,self.state[self.state_name[next_s]],r)
                    self.loss=loss
                    with tf.GradientTape() as tape:
                        _gradient=tape.gradient(loss,self.value_p)
                        for i in range(len(_gradient)):
                            g[i]=g[i]+(self.alpha*g[i]+(1-self.alpha)*_gradient[i]**2)
                            gradient[i]=gradient[i]+self.lr*_gradient[i]/tf.math.sqrt(g[i]+self.epsilon)
                    s=next_s
                    self.T+=1
                    t+=1
                    if self.T>self.Tmax:
                        return
                    if self.T%self.It==0:
                        self.update_parameter()
                    elif t%self.Ia==0 or end:
                        for i in range(len(self.value_p)):
                            self.value_p[i]=self.value_p[i]+gradient[i]
                            gradient=[0 for _ in range(len(self.value_p))]
                            g=[0 for _ in range(len(self.value_p))]
            self.total_episode+=1
            if self.save_episode==True:
                self.episode.append(episode)
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump(self.value_p,parameter_file)
        return
    
    
    def save_e(self,path):
        episode_file=open(path+'.dat','wb')
        pickle.dump(self.episode,episode_file)
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'\save.dat','wb')
            path=path+'\save.dat'
            index=path.rfind('\\')
            episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
        else:
            output_file=open(path+'\save-{0}.dat'.format(i+1),'wb')
            path=path+'\save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            episode_file=open(path.replace(path[index+1:],'episode-{0}.dat'.format(i+1)),'wb')
        pickle.dump(self.episode,episode_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.T,output_file)
        pickle.dump(self.Tmax,output_file)
        pickle.dump(self.It,output_file)
        pickle.dump(self.Ia,output_file)
        pickle.dump(self.alpha,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.total_episode,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,e_path):
        input_file=open(s_path,'rb')
        episode_file=open(e_path,'rb')
        self.episode=pickle.load(episode_file)
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.T=pickle.load(input_file)
        self.Tmax=pickle.load(input_file)
        self.It=pickle.load(input_file)
        self.Ia=pickle.load(input_file)
        self.alpha=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        input_file.close()
        return
