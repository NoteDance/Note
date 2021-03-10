import tensorflow as tf
import numpy as np
import pickle
import time


class Q_learning:
    def __init__(self,net,net_p,state,state_name,action_name,exploration_space,epsilon=None,discount=None,episode_step=None,optimizer=None,lr=None,save_episode=True):
        self.net=net
        self.net_p=net_p
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.action_len=len(self.action_name)
        self.epsilon=epsilon
        self.discount=discount
        self.episode_step=episode_step
        self.optimizer=optimizer
        self.lr=lr
        self.save_episode=save_episode
        self.opt_flag=False
        self.epi_num=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
       
        
    def init(self,dtype=np.int32):
        self.t3=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            self.action_one=np.concatenate((self.action_prob,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_one=np.ones(len(self.action_name),dtype=dtype)
        self.t4=time.time()
        return
    
    
    def set_up(self,net_p=None,epsilon=None,discount=None,episode_step=None,optimizer=None,lr=None,init=True):
        if net_p!=None:
            self.net_p=net_p
        if epsilon!=None:
            self.epsilon=epsilon
        if discount!=None:
            self.discount=discount
        if episode_step!=None:
            self.episode_step=episode_step
        if optimizer!=None:
            self.optimizer=optimizer
        if lr!=None:
            self.lr=lr
            self.optimizer.lr=lr
        if init==True:
            self.episode=[]
            self.epi_num=0
            self.episode_num=0
            self.total_episode=0
            self.time=0
            self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_action=np.argmax(self.net(self.state[self.state_name[s]]))
        action_prob[best_action]+=1-self.epsilon
        return action_prob
    
    
    def loss(self,s,a,next_s,r):
        return (r+self.discount*tf.reduce_max(self.net(self.state[next_s]))-self.net(self.state[s])[a])**2
    
    
    def learn(self,episode_num,path=None,one=True):
        if self.lr!=None:
            self.optimizer.lr=self.lr
        for i in range(episode_num):
            loss=0
            episode=[]
            s=int(np.random.uniform(0,len(self.state_name)))
            if self.episode_step==None:
                while True:
                    t1=time.time()
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    with tf.GradientTape() as tape:
                        loss+=self.loss(s,a,next_s,r)
                    gradient=tape.gradient(1/2*loss,self.net_p)
                    if self.opt_flag==True:
                        self.optimizer(gradient,self.net_p)
                    else:
                        self.optimizer.apply_gradients(zip(gradient,self.net_p))
                    loss=loss.numpy()
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                        t2=time.time()
                        self.time+=(t2-t1)
                        break
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    s=next_s
                    t2=time.time()
                    self.time+=(t2-t1)
            else:
                for _ in range(self.episode_step):
                    t1=time.time()
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    with tf.GradientTape() as tape:
                        loss+=self.loss(s,a,next_s,r)
                    gradient=tape.gradient(1/2*loss,self.net_p)
                    if self.opt_flag==True:
                        self.optimizer(gradient,self.net_p)
                    else:
                        self.optimizer.apply_gradients(zip(gradient,self.net_p))
                    loss=loss.numpy()
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                        t2=time.time()
                        self.time+=(t2-t1)
                        break
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    s=next_s
                    t2=time.time()
                    self.time+=(t2-t1)
            if episode_num%10!=0:
                d=episode_num-episode_num%10
                d=int(d/10)
            else:
                d=episode_num/10
            if d==0:
                d=1
            if i%d==0:
                print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                if path!=None and i%episode_num*2==0:
                    self.save(path,i,one)
            self.epi_num+=1
            self.total_episode+=1
        if path!=None:
            self.save(path)
        if self.save_episode==True:
            self.episode.append(episode)
        if self.time<0.5:
            self.time=int(self.time+(self.t4-self.t3))
        else:
            self.time=int(self.time+(self.t4-self.t3))+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(loss))
        print('time:{0}s'.format(self.time))
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump(self.net_p,parameter_file)
        parameter_file.close()
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
            episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
        else:
            output_file=open(path+'\save-{0}.dat'.format(i+1),'wb')
            path=path+'\save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            episode_file=open(path.replace(path[index+1:],'episode-{0}.dat'.format(i+1)),'wb')
        self.episode_num=self.epi_num
        pickle.dump(self.episode,episode_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.opt_flag,output_file)
        pickle.dump(self.state_one,output_file)
        pickle.dump(self.episode_num,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        episode_file.close()
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
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.opt_flag=pickle.load(input_file)
        self.state_one=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        episode_file.close()
        return
