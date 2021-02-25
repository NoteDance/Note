import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class DQN:
    def __init__(self,value_net,value_p,target_p,state,state_name,action_name,exploration_space,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,optimizer=None,lr=None,pool_net=True,save_episode=True):
        self.value_net=value_net
        self.value_p=value_p
        self.target_p=target_p
        self.state_pool=[]
        self.action_pool=[]
        self.next_state_pool=[]
        self.reward_pool=[]
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.action_len=len(self.action_name)
        self.epsilon=[]
        self.discount=discount
        self.episode_step=episode_step
        self.pool_size=pool_size
        self.batch=batch
        self.update_step=update_step
        self.optimizer=optimizer
        self.lr=lr
        self.t=0
        self.t_counter=0
        self.one_list=[]
        self.index_list=[]
        self.p=None
        self.flish_list=[]
        self.pool_net=pool_net
        self.save_episode=save_episode
        self.loss=[]
        self.loss_list=[]
        self.opt_flag==False
        self.a=[]
        self.epi_num=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
        
        
    def init(self,dtype=np.int32):
        t3=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_one=np.ones(len(self.action_name),dtype=dtype)
        self.index=np.arange(self.batch,dtype=np.int8)
        t4=time.time()
        self.time+=t4-t3
        return
    
    
    def set_up(self,value_p=None,target_p=None,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,optimizer=None,lr=None,init=True):
        if value_p!=None:
            self.value_p=value_p
            self.target_p=target_p
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
        if init==True:
            self.t=0
            self.t_counter=0
            self.one_list=[]
            self.index_list=[]
            self.p=None
            self.flish_list=[]
            self.pool_net=True
            self.episode=[]
            self.epsilon=[]
            self.state_pool=None
            self.action_pool=None
            self.next_state_pool=None
            self.reward_pool=None
            self.loss=[]
            self.loss_list=[]
            self.a=[]
            self.epi_num=0
            self.episode_num=0
            self.total_episode=0
            self.time=0
            self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one,epsilon):
        action_prob=action_one
        action_prob=action_prob*epsilon/len(action_one)
        best_a=np.argmax(self.value_net(self.state[self.state_name[s]]))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def update_parameter(self):
        for i in range(len(self.value_p)):
            self.target_p[i]=self.value_p[i].copy()
        return
    
    
    def _loss(self,s,a,next_s,r):
        if len(self.state_pool)<self.batch:
            return tf.reduce_mean(((r+self.discount*tf.reduce_max(self.value_net(next_s,self.target_p),axis=-1))-self.value_net(s,self.value_p)[np.arange(len(a)),a])**2)
        else:
            return tf.reduce_mean(((r+self.discount*tf.reduce_max(self.value_net(next_s,self.target_p),axis=-1))-self.value_net(s,self.value_p)[self.index,a])**2)
    
    
    def explore(self,s,epsilon,i):
        episode=[]
        action_prob=self.epsilon_greedy_policy(s,self.action_one,epsilon)
        a=np.random.choice(self.action,p=action_prob)
        next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
        if self.pool_net==True:
            flag=np.random.randint(0,2)
            while True:
                index=np.random.choice(self.index_list,p=self.p)
                if index in self.finish_list:
                    continue
                else:
                    break
        if self.pool_net==True and flag==1 and self.state_pool[index]!=None and len(self.state_pool)==self.t_counter:
            self.state_pool[index]=tf.concat([self.state_pool[index],tf.expand_dims(self.state[self.state_name[s]],axis=0)])
            self.action_pool[index]=tf.concat([self.action_pool[index],tf.expand_dims(a,axis=0)])
            self.next_state_pool[index]=tf.concat([self.next_state_pool[index],tf.expand_dims(self.state[self.state_name[next_s]],axis=0)])
            self.reward_pool[index]=tf.concat([self.reward_pool[index],tf.expand_dims(r,axis=0)])
        else:
            if self.state_pool[i]==None:
                self.state_pool[i]=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                self.action_pool[i]=tf.expand_dims(a,axis=0)
                self.next_state_pool[i]=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                self.reward_pool[i]=tf.expand_dims(r,axis=0)
            else:
                self.state_pool[i]=tf.concat([self.state_pool[i],tf.expand_dims(self.state[self.state_name[s]],axis=0)])
                self.action_pool[i]=tf.concat([self.action_pool[i],tf.expand_dims(a,axis=0)])
                self.next_state_pool[i]=tf.concat([self.next_state_pool[i],tf.expand_dims(self.state[self.state_name[next_s]],axis=0)])
                self.reward_pool[i]=tf.concat([self.reward_pool[i],tf.expand_dims(r,axis=0)])
        if len(self.state_pool[i])>self.pool_size:
            self.state_pool[i]=self.state_pool[i][1:]
            self.action_pool[i]=self.action_pool[i][1:]
            self.next_state_pool[i]=self.next_state_pool[i][1:]
            self.reward_pool[i]=self.reward_pool[i][1:]
        if end:
            if self.save_episode==True:
                episode.append([self.state_name[s],self.action_name[a],r,end])
        elif self.save_episode==True:
            episode.append([self.state_name[s],self.self.action_name[a],r])
        if self.save_episode==True:
            self.episode.append(episode)
        self.epi_num+=1
        return next_s,end
    
    
    def _learn(self,i):
        if len(self.state_pool)<self.batch:
            with tf.GradientTape() as tape:
                self.loss[i]=self._loss(self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i])
            gradient=tape.gradient(self.loss[i],self.value_p)
            if self.opt_flag==True:
                self.optimizer(gradient,self.value_p)
            else:
                self.optimizer.apply_gradients(zip(gradient,self.value_p))
            if self.a%self.update_step==0:
                self.update_parameter()
        else:
            batches=int((len(self.state_pool[i])-len(self.state_pool[i])%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if self.pool_net!=True:
                train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[i],self.action_pool[i],self.next_state_pool[i],)).shuffle(len(self.state_pool[i])).batch(self.batch)
            self.loss[i]=0
            if self.pool_net==True:
                for j in range(batches):
                    index1=j*self.batch
                    index2=(j+1)*self.batch
                    state_batch=self.state_pool[i][index1:index2]
                    action_batch=self.action_pool[i][index1:index2]
                    next_state_batch=self.next_state_pool[i][index1:index2]
                    reward_batch=self.reward_pool[i][index1:index2]
                    with tf.GradientTape() as tape:
                        batch_loss=self._loss(state_batch,action_batch,next_state_batch,reward_batch)
                    gradient=tape.gradient(batch_loss,self.value_p)
                    if self.opt_flag==True:
                        self.optimizer(gradient,self.value_p)
                    else:
                        self.optimizer.apply_gradients(zip(gradient,self.value_p))
                    self.loss[i]+=batch_loss
            else:
                for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
                    with tf.GradientTape() as tape:
                        batch_loss=self._loss(state_batch,action_batch,next_state_batch,reward_batch)
                    gradient=tape.gradient(batch_loss,self.value_p)
                    if self.opt_flag==True:
                        self.optimizer(gradient,self.value_p)
                    else:
                        self.optimizer.apply_gradients(zip(gradient,self.value_p))
                    self.loss[i]+=batch_loss
            if self.a[i]%self.update_step==0:
                self.update_parameter()
            if len(self.state_pool)%self.batch!=0:
                self.loss[i]=self.loss[i].numpy()/self.batches+1
            elif len(self.state_pool)<self.batch:
                self.loss[i]=self.loss[i].numpy()
            else:
                self.loss[i]=self.loss[i].numpy()/self.batches
        return
    
    
    def learn(self,episode_num,epsilon,i):
        self.t+=1
        self.t_counter+=1
        self.one_list.append(1)
        self.index_list.append(i)
        self.p=np.array(self.one_list,dtype=np.float16)/self.t_counter
        self.a.append(0)
        self.loss.append(0)
        if len(self.state_pool)==i-1:
            self.state_pool.append(None)
            self.action_pool.append(None)
            self.next_state_pool.append(None)
            self.reward_pool.append(None)
            self.epsilon.append(epsilon)
        for _ in range(episode_num):
            s=int(np.random.uniform(0,len(self.state_name)))
            if self.episode_step==None:
                while True:
                    self.a[i]+=1
                    next_s,end=self.explore(s,self.epsilon[i],i)
                    s=next_s
                    self._learn(i)
                    if end:
                        break
            else:
                for _ in range(self.episode_step):
                    self.a[i]+=1
                    next_s,end=self.explore(s,self.epsilon[i],i)
                    s=next_s
                    self._learn(i)
                    if end:
                        break
        self.t_counter-=1
        self.one_list[i]=0
        self.p=np.array(self.one_list,dtype=np.float16)/self.t_counter
        self.finish_list.append(i)
        self.state_pool[i]=tf.expand_dims(self.state_pool[i][0],axis=0)
        self.action_pool[i]=tf.expand_dims(self.action_pool[i][0],axis=0)
        self.next_state_pool[i]=tf.expand_dims(self.next_state_pool[i][0],axis=0)
        self.reward_pool[i]=tf.expand_dims(self.reward_pool[i][0],axis=0)
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
        self.thread=self.t
        pickle.dump(self.episode,episode_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.index,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.thread,output_file)
        pickle.dump(self.t_counter,output_file)
        pickle.dump(self.one_list,output_file)
        pickle.dump(self.index_list,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.finish_list,output_file)
        pickle.dump(self.pool_net,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.opt_flag,output_file)
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
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.index=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.thread=pickle.load(input_file)
        self.t_counter=pickle.load(input_file)
        self.one_list=pickle.load(input_file)
        self.index_list=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.finish_list=pickle.load(input_file)
        self.pool_net=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.opt_flag=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
