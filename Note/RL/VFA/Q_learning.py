import tensorflow as tf
import numpy as np
import pickle
import time


class Q_learning:
    def __init__(self,net,net_p,state,state_name,action_name,search_space,epsilon=None,discount=None,episode_step=None,optimizer=None,lr=None,save_episode=True):
        self.net=net
        self.net_p=net_p
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.search_space=search_space
        self.action_len=len(self.action_name)
        self.epsilon=epsilon
        self.discount=discount
        self.episode_step=episode_step
        self.lr=lr
        self.optimizer=optimizer
        self.save_episode=save_episode
        self.loss=0
        self.opt_flag=False
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
       
        
    def init(self,dtype=np.int32):
        t3=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate(self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len)
            self.action_p=np.concatenate(self.action_prob,np.ones(len(self.action_name)-self.action_len,dtype=dtype))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_p=np.ones(len(self.action_name),dtype=dtype)
        t4=time.time()
        self.time+=t4-t3
        return
    
    
    def epsilon_greedy_policy(self,s,action_p):
        action_prob=action_p
        action_prob=action_prob*self.epsilon/np.sum(action_p)
        best_action=np.argmax(self.predict_net(self.state[self.state_name[s]]).numpy())
        action_prob[best_action]+=1-self.epsilon
        return action_prob
    
    
    def _loss(self,s,a,next_s,r):
        return (r+self.discount*tf.reduce_max(self.net(self.state[next_s]))-self.net(self.state[s])[a])**2
    
    
    def episode(self):
        episode=[]
        s=int(np.random.uniform(0,len(self.state_name)))
        if self.episode_step==None:
            while True:
                action_prob=self.epsilon_greedy_policy(s,self.action_p)
                a=np.random.choice(self.action,p=action_prob)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                if end:
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.self.action_name[a],r])
                self.loss+=self._loss(s,a,next_s,r)
                s=next_s
        else:
            for _ in range(self.episode_step):
                action_prob=self.epsilon_greedy_policy(s,self.action_p)
                a=np.random.choice(self.action,p=action_prob)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                if end:
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.self.action_name[a],r])
                self.loss+=self._loss(s,a,next_s,r)
                s=next_s
        if self.save_episode==True:
            self.episode.append(episode)
        return
        
    
    def learn(self):
        with tf.GradientTape() as tape:
            gradient=tape.gradient(1/2*self.loss,self.net_p)
            if self.opt_flag==True:
                self.optimizer(gradient,self.net_p)
            else:
                self.optimizer.apply_gradients(zip(gradient,self.net_p))
        self.loss=self.loss.numpy()
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'save.dat','wb')
            path=path+'save.dat'
            index=path.rfind('\\')
            episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
        else:
            output_file=open(path+'save-{0}.dat'.format(i+1),'wb')
            path=path+'save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            episode_file=open(path.replace(path[index+1:],'episode-{0}.dat'.format(i+1)),'wb')
        pickle.dump(self.episode,episode_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_p,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.opt_flag,output_file)
        pickle.dump(self.state_one,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,e_path):
        input_file=open(s_path,'rb')
        episode_file=open(e_path,'rb')
        self.episode=pickle.load(episode_file)
        self.action_len=pickle.load(input_file)
        if self.action_len==len(self.action_name):
            self.action=pickle.load(input_file)
            self.action_p=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.opt_flag=pickle.load(input_file)
        self.state_one=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
