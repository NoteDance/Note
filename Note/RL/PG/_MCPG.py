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
        self.lr=lr
        self.optimizer=optimizer
        self.save_episode=save_episode
        self.opt_flag=False
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def init(self,dtype=np.int32):
        self.t3=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate(self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len)
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
        self.t4=time.time()
        return
        
        
    def loss(self,output,G):
        return tf.log(output)*G
    
    
    def episode(self,s,action):
        G=0
        loss=0
        episode=[]
        if self.episode_step==None:
            output=self.policy_net(self.state[s])
            while True:
                output=self.policy_net(self.state[s])
                a=np.random.choice(action,output)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                if end:
                    self.reward_list.append(G)
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],r])
                G+=r
                loss+=self.loss(output,G)
                s=next_s
        else:
            output=self.policy_net(self.state[s])
            for _ in range(self.episode_step):
                output=self.policy_net(self.state[s])
                a=np.random.choice(action,output)
                next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                if end:
                    self.reward_list.append(G)
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state_name[s],self.action_name[a],r])
                G+=r
                loss+=self.loss(output,G)
                s=next_s
            self.reward_list.append(G)
        if self.save_episode==True:
            self.episode.append(episode)
        return loss
    
    
    def learn(self,episode_num,path=None,one=True):
        for i in range(episode_num):
            t1=time.time()
            s=int(np.random.uniform(0,len(self.state_name)))
            loss=self.episode(s,self.action)
            with tf.GradientTape() as tape:
                gradient=tape.gradient(loss,self.net_p)
                if self.opt_flag==True:
                    self.optimizer(gradient,self.net_p)
                else:
                    self.optimizer.apply_gradients(zip(gradient,self.net_p))
            t2=time.time()
            self.time+=(t2-t1)
            t2=time.time()
            self.time+=(t2-t1)
            if episode_num%10!=0:
                temp=episode_num-episode_num%10
                temp=int(temp/10)
            else:
                temp=episode_num/10
            if temp==0:
                temp=1
            if i%temp==0:
                print('episode num:{0}   max reward:{1}'.format(i+1,max(self.reward_list)))
                if path!=None and i%episode_num*2==0:
                    self.save(path,i,one)
            self.episode_num+=1
            self.total_episode+=1
            if self.reward_min!=None and max(self.reward_list)>=self.reward_min:
               break 
        if self.time<0.5:
            self.time=int(self.time+(self.t4-self.t3))
        else:
            self.time=int(self.time+(self.t4-self.t3))+1
        self.total_time+=self.time
        print()
        print('last max reward:{0}'.format(max(self.reward_list)))
        print('time:{0}s'.format(self.time))
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
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
    
    
    def restore(self,path):
        input_file=open(path,'rb')
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
