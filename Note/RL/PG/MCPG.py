import tensorflow as tf
import numpy as np
import pickle
import time


class MCPG:
    def __init__(self,policy_net,net_p,state,state_name,action_name,search_space,epsilon=None,discount=None,reward_min=None,episode_step=None,optimizer=None,lr=None,save_episode=True):
        self.policy_net=policy_net
        self.net_p=net_p
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.search_space=search_space
        self.action_len=len(self.action_name)
        self.reward_list=[]
        self.epsilon=epsilon
        self.discount=discount
        self.reward_min=reward_min
        self.episode_step=episode_step
        self.optimizer=optimizer
        self.lr=lr
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
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
        t4=time.time()
        self.time+=t4-t3
        return
        
        
    def _loss(self,output,G):
        return tf.log(output)*G
    
    
    def episode(self):
        G=0
        episode=[]
        s=int(np.random.uniform(0,len(self.state_name)))
        if self.episode_step==None:
            output=self.policy_net(self.state[s])
            while True:
                output=self.policy_net(self.state[s])
                a=np.random.choice(self.action,output)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                if end:
                    self.reward_list.append(G)
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],r])
                G+=r
                self.loss+=self._loss(output,G)
                s=next_s
        else:
            output=self.policy_net(self.state[s])
            for _ in range(self.episode_step):
                output=self.policy_net(self.state[s])
                a=np.random.choice(self.action,output)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                if end:
                    self.reward_list.append(G)
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],r])
                G+=r
                self.loss+=self._loss(output,G)
                s=next_s
            self.reward_list.append(G)
        if self.save_episode==True:
            self.episode.append(episode)
        return
    
    
    def learn(self):
        with tf.GradientTape() as tape:
            gradient=tape.gradient(self.loss,self.net_p)
            if self.opt_flag==True:
                self.optimizer(gradient,self.net_p)
            else:
                self.optimizer.apply_gradients(zip(gradient,self.net_p))
        if self.reward_min!=None and max(self.reward_list)>=self.reward_min:
           return 
        return
    
    
    def save_p(self,path):
        output_file=open(path+'.dat','wb')
        pickle.dump(self.net_p,output_file)
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
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.reward_min,output_file)
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
        self.reward_list=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.reward_min=pickle.load(input_file)
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
