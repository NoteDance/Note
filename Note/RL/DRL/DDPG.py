import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class DDPG:
    def __init__(self,value_net,actor_net,value_target_p,value_p,actor_target_p,actor_p,state,state_name,action_name,exploration_space,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,optimizer=None,lr=None,tau=0.001,save_episode=True):
        self.value_net=value_net
        self.actor_net=actor_net
        self.value_p=value_p
        self.value_target_p=value_target_p
        self.actor_p=actor_p
        self.actor_target_p=actor_target_p
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.discount=discount
        self.episode_step=episode_step
        self.pool_size=pool_size
        self.batch=batch
        self.update_step=update_step
        self.optimizer=optimizer
        self.lr=lr
        self.tau=tau
        self.save_episode=save_episode
        self.loss_list=[]
        self.opt_flag==False
        self.epi_num=0
        self.episode_num=0
        self.a=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def set_up(self,value_p=None,value_target_p=None,actor_p=None,actor_target_p=None,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,optimizer=None,lr=None,tau=0.001,init=True):
        if value_p!=None:
            self.value_p=value_p
            self.value_target_p=value_target_p
            self.actor_p=actor_p
            self.actor_target_p=actor_target_p
        if discount!=None:
            self.discount=discount
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
            self.index=np.arange(self.batch,dtype=np.int8)
        if update_step!=None:
            self.update_step=update_step
        if optimizer!=None:
            self.optimizer=optimizer
        if lr!=None:
            self.lr=lr
        if tau!=None:
            self.tau=tau
        if init==True:
            self.episode=[]
            self.state_pool=None
            self.action_pool=None
            self.next_state_pool=None
            self.reward_pool=None
            self.loss_list=[]
            self.a=0
            self.epi_num=0
            self.episode_num=0
            self.total_episode=0
            self.time=0
            self.total_time=0
        return
        
        
    def OU(self):
        
        
    
    
    def sampled_gradient(self,value_gradient,actor_gradient):
        for i in range(len(value_gradient)):
           actor_gradient[i]=tf.reduce_sum(value_gradient[i]*actor_gradient[i],axis=0)/len(value_gradient[i])
        return actor_gradient
    
    
    def update_parameter(self):
        for i in range(len(self.value_predict_p)):
            self.value_target_p[i]=self.tau*self.value_target_p[i]+(1-self.tau)*self.value_p[i]
        for i in range(len(self.actor_p)):
            self.actor_target_p[i]=self.tau*self.actor_target_p[i]+(1-self.tau)*self.actor_p[i]
        return
    
    
    def _loss(self,value,next_s,r):
        return tf.reduce_mean(((r+self.discount*self.Q_target_net(next_s,self.actor_net(next_s,self.actor_target_p),self.value_target_p))-value)**2)
    
    
    def explore(self,episode_num):
        episode=[]
        s=int(np.random.uniform(0,len(self.state_name)))
        for _ in range(episode_num):
            if self.episode_step==None:
                while True:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                        break
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    self.a+=1
                    if self.state_pool==None:
                        self.state_pool=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                        self.action_pool=tf.expand_dims(a,axis=0)
                        self.next_state_pool=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                        self.reward_pool=tf.expand_dims(r,axis=0)
                    else:
                        self.state_pool=tf.concat([self.state_pool,tf.expand_dims(self.state[self.state_name[s]],axis=0)])
                        self.action_pool=tf.concat([self.action_pool,tf.expand_dims(a,axis=0)])
                        self.next_state_pool=tf.concat([self.next_state_pool,tf.expand_dims(self.state[self.state_name[next_s]],axis=0)])
                        self.reward_pool=tf.concat([self.reward_pool,tf.expand_dims(r,axis=0)])
                    if len(self.state_pool)>self.pool_size:
                        self.state_pool=self.state_pool[1:]
                        self.action_pool=self.action_pool[1:]
                        self.next_state_pool=self.next_state_pool[1:]
                        self.reward_pool=self.reward_pool[1:]
                    s=next_s
            else:
                for _ in range(self.episode_step):
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                        break
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    self.a+=1
                    if self.state_pool==None:
                        self.state_pool=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                        self.action_pool=tf.expand_dims(a,axis=0)
                        self.next_state_pool=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                        self.reward_pool=tf.expand_dims(r,axis=0)
                    else:
                        self.state_pool=tf.concat([self.state_pool,tf.expand_dims(self.state[self.state_name[s]],axis=0)])
                        self.action_pool=tf.concat([self.action_pool,tf.expand_dims(a,axis=0)])
                        self.next_state_pool=tf.concat([self.next_state_pool,tf.expand_dims(self.state[self.state_name[next_s]],axis=0)])
                        self.reward_pool=tf.concat([self.reward_pool,tf.expand_dims(r,axis=0)])
                    if len(self.state_pool)>self.pool_size:
                        self.state_pool=self.state_pool[1:]
                        self.action_pool=self.action_pool[1:]
                        self.next_state_pool=self.next_state_pool[1:]
                        self.reward_pool=self.reward_pool[1:]
                    s=next_s
            if self.save_episode==True:
                self.episode.append(episode)
            self.epi_num+=1
        return
    
    
    def learn(self):
        self.loss=0
        if len(self.state_pool)<self.batch:
            value=self.value_net(self.state_pool,self.action_pool,self.value_p)
            loss=self._loss(value,self.next_state_pool,self.reward_pool)
            with tf.GradientTape() as tape:
                gradient=tape.gradient(self.loss,self.value_p)
                value_gradient=tape.gradient(value,self.action_pool)
                actor_gradient=tape.gradient(self.action_pool,self.state_pool)
                actor_gradient=self.sampled_gradient(value_gradient,actor_gradient)
                if self.opt_flag==True:
                    self.optimizer(gradient,self.value_p)
                else:
                    self.optimizer.apply_gradients(zip(gradient,self.value_p))
                for i in range(len(self.actor_p)):
                    self.actor_p[i]=self.actor_p[i]-actor_gradient[i]
                if self.a%self.update_step==0:
                    self.update_parameter()
        else:
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            for j in range(batches):
                index1=j*self.batch
                index2=(j+1)*self.batch
                state_batch=self.state_pool[index1:index2]
                action_batch=self.action_pool[index1:index2]
                next_state_batch=self.next_state_pool[index1:index2]
                reward_batch=self.reward_pool[index1:index2]
                value=self.value_net(state_batch,action_batch,self.value_p)
                batch_loss=self._loss(value,next_state_batch,reward_batch)
                with tf.GradientTape() as tape:
                    gradient=tape.gradient(batch_loss,self.value_p)
                    value_gradient=tape.gradient(value,action_batch)
                    actor_gradient=tape.gradient(action_batch,state_batch)
                    actor_gradient=self.sampled_gradient(value_gradient,actor_gradient)
                    if self.opt_flag==True:
                        self.optimizer(gradient,self.value_p)
                    else:
                        self.optimizer.apply_gradients(zip(gradient,self.value_p))
                    for i in range(len(self.actor_p)):
                        self.actor_p[i]=self.actor_p[i]-actor_gradient[i]
                self.loss+=batch_loss
            if len(self.state_pool)%self.batch!=0:
                batches+=1
                index1=batches*self.batch
                index2=self.batch-(self.shape0-batches*self.batch)
                state_batch=tf.concat([self.state_pool[index1:],self.state_pool[:index2]])
                action_batch=tf.concat([self.action_pool[index1:],self.action_pool[:index2]])
                next_state_batch=tf.concat([self.next_state_pool[index1:],self.next_state_pool[:index2]])
                reward_batch=tf.concat([self.reward_pool[index1:],self.reward_pool[:index2]])
                value=self.value_net(state_batch,action_batch,self.value_p)
                batch_loss=self._loss(value,next_state_batch,reward_batch)
                with tf.GradientTape() as tape:
                    gradient=tape.gradient(batch_loss,self.value_p)
                    value_gradient=tape.gradient(value,action_batch)
                    actor_gradient=tape.gradient(action_batch,state_batch)
                    actor_gradient=self.sampled_gradient(value_gradient,actor_gradient)
                    if self.opt_flag==True:
                        self.optimizer(gradient,self.value_p)
                    else:
                        self.optimizer.apply_gradients(zip(gradient,self.value_p))
                    for i in range(len(self.actor_p)):
                        self.actor_p[i]=self.actor_p[i]-actor_gradient[i]
                self.loss+=batch_loss
            if len(self.state_pool)%self.batch!=0:
                self.loss=self.loss.numpy()/self.batches+1
            elif len(self.state_pool)<self.batch:
                self.loss=self.loss.numpy()
            else:
                self.loss=self.loss.numpy()/self.batches
        if self.a%self.update_step==0:
            self.update_parameter()
        return
    
    
    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_episode),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump(self.value_p,parameter_file)
        pickle.dump(self.actor_p,parameter_file)
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
        self.episode_num=self.epi_num
        pickle.dump(self.episode,episode_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.tau,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.opt_flag,output_file)
        pickle.dump(self.a,output_file)
        pickle.dump(self.episode_num,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,e_path):
        input_file=open(s_path,'rb')
        episode_file=open(e_path,'rb')
        self.episode=pickle.load(episode_file)
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.tau=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.opt_flag=pickle.load(input_file)
        self.a=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
