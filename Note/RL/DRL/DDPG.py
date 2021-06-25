import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class DDPG:
    def __init__(self,value_net,actor_net,state,state_name,variance,exploration_space,pool_net=True,save_episode=True):
        self.value_net=value_net
        self.actor_net=actor_net
        self.value_p=None
        self.value_target_p=None
        self.actor_p=None
        self.actor_target_p=None
        self.state_pool=[]
        self.action_pool=[]
        self.next_state_pool=[]
        self.reward_pool=[]
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.variance=variance
        self.exploration_space=exploration_space
        self.discount=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.optimizer=None
        self.lr=None
        self.tau=None
        self.end_loss=None
        self.t=0
        self.t_counter=0
        self.one_list=[]
        self.index_list=[]
        self.use_flag=[]
        self.p=None
        self.flish_list=[]
        self.pool_net=pool_net
        self.save_episode=save_episode
        self.loss=[]
        self.loss_list=[]
        self.opt_flag==False
        self.epi_num=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def set_up(self,value_p=None,value_target_p=None,actor_p=None,actor_target_p=None,discount=None,episode_step=None,pool_size=None,batch=None,optimizer=None,lr=None,tau=0.001,end_loss=None,init=False):
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
        if optimizer!=None:
            self.optimizer=optimizer
        if lr!=None:
            self.lr=lr
            self.optimizer.lr=lr
        elif self.lr!=None:
            self.optimizer.lr=self.lr
        if tau!=None:
            self.tau=tau
        if end_loss!=None:
            self.end_loss=end_loss
        if init==True:
            self.t=0
            self.t_counter=0
            self.one_list=[]
            self.index_list=[]
            self.use_flag=[]
            self.p=None
            self.flish_list=[]
            self.pool_net=True
            self.episode=[]
            self.state_pool=[]
            self.action_pool=[]
            self.next_state_pool=[]
            self.reward_pool=[]
            self._TD=[]
            self.loss=[]
            self.loss_list=[]
            self.epi_num=0
            self.episode_num=0
            self.total_episode=0
            self.time=0
            self.total_time=0
        return
        
        
    def OU(self):
        return np.random.normal(0,self.variance)
    
    
    def update_parameter(self):
        for i in range(len(self.value_predict_p)):
            self.value_target_p[i]=self.tau*self.value_target_p[i]+(1-self.tau)*self.value_p[i]
        for i in range(len(self.actor_p)):
            self.actor_target_p[i]=self.tau*self.actor_target_p[i]+(1-self.tau)*self.actor_p[i]
        return
    
    
    def TD(self,value,s,next_s,r):
        return tf.reduce_mean((r+self.discount*self.value_net(next_s,self.value_target_p)-value)**2)
        
    
    def explore(self,s,i):
        a=self.actor_net(self.state[self.state_name[s]],self.actor_p).numpy()+self.OU()
        next_s,r,end=self.exploration_space(self.state_name[s],a)
        if self.pool_net==True:
            flag=np.random.randint(0,2)
            while True:
                index=np.random.choice(self.index_list,p=self.p)
                if index in self.finish_list or self.use_flag[i]==True:
                    continue
                else:
                    break
        if self.pool_net==True and flag==1 and self.state_pool[index]!=None and index in self.finish_list and self.use_flag[i]==False:
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
                episode=[self.state_name[s],a,self.state_name[next_s],r,end]
        elif self.save_episode==True:
            episode=[self.state_name[s],a,self.state_name[next_s],r]
        self.epi_num+=1
        return next_s,end,episode
    
    
    def _learn(self,i):
        self.use_flag[i]=True
        if len(self.state_pool[i])<self.batch:
            with tf.GradientTape() as tape:
                value=self.value_net(self.state_pool[i],self.value_p)
                self._TD[i]=self.TD(value,self.state_pool[i],self.next_state_pool[i],self.reward_pool[i])
            value_gradient=tape.gradient(self._TD[i],self.value_p)
            actor_gradient=self._TD[i]*tape.gradient(tf.math.log(self.action_pool[i]),self.actor_p)
            if self.opt_flag==True:
                self.optimizer(value_gradient,self.value_p)
                self.optimizer(actor_gradient,self.actor_p)
            else:
                self.optimizer.apply_gradients(zip(value_gradient,self.value_p))
                self.optimizer.apply_gradients(zip(actor_gradient,self.value_p))
            self.loss[i]=self._TD[i]
            self.update_parameter()
        else:
            self.loss[i]=0
            batches=int((len(self.state_pool[i])-len(self.state_pool[i])%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if self.pool_net==True:
                for j in range(batches):
                    index1=j*self.batch
                    index2=(j+1)*self.batch
                    state_batch=self.state_pool[i][index1:index2]
                    action_batch=self.action_pool[i][index1:index2]
                    next_state_batch=self.next_state_pool[i][index1:index2]
                    reward_batch=self.reward_pool[i][index1:index2]
                    with tf.GradientTape() as tape:
                        value=self.value_net(state_batch,self.value_p)
                        self._TD[i]=self.TD(value,state_batch,next_state_batch,reward_batch)
                    value_gradient=tape.gradient(self._TD[i],self.value_p)
                    actor_gradient=self._TD[i]*tape.gradient(tf.math.log(action_batch),self.actor_p)
                    if self.opt_flag==True:
                        self.optimizer(value_gradient,self.value_p)
                        self.optimizer(actor_gradient,self.actor_p)
                    else:
                        self.optimizer.apply_gradients(zip(value_gradient,self.value_p))
                        self.optimizer.apply_gradients(zip(actor_gradient,self.value_p))
                    self.loss[i]+=self._TD[i]
                    self.update_parameter()
                if len(self.state_pool)%self.batch!=0:
                    batches+=1
                    index1=batches*self.batch
                    index2=self.batch-(self.shape0-batches*self.batch)
                    state_batch=tf.concat([self.state_pool[i][index1:],self.state_pool[i][:index2]])
                    action_batch=tf.concat([self.action_pool[i][index1:],self.action_pool[i][:index2]])
                    next_state_batch=tf.concat([self.next_state_pool[i][index1:],self.next_state_pool[i][:index2]])
                    reward_batch=tf.concat([self.reward_pool[i][index1:],self.reward_pool[i][:index2]])
                    with tf.GradientTape() as tape:
                        value=self.value_net(state_batch,self.value_p)
                        self._TD[i]=self.TD(value,state_batch,next_state_batch,reward_batch)
                    value_gradient=tape.gradient(self._TD[i],self.value_p)
                    actor_gradient=self._TD[i]*tape.gradient(tf.math.log(action_batch),self.actor_p)
                    if self.opt_flag==True:
                        self.optimizer(value_gradient,self.value_p)
                        self.optimizer(actor_gradient,self.actor_p)
                    else:
                        self.optimizer.apply_gradients(zip(value_gradient,self.value_p))
                        self.optimizer.apply_gradients(zip(actor_gradient,self.value_p))
                    self.loss[i]+=self._TD[i]
                    self.update_parameter()
            else:
                train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i])).shuffle(len(self.state_pool[i])).batch(self.batch)
                for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
                    with tf.GradientTape() as tape:
                        value=self.value_net(state_batch,self.value_p)
                        self._TD[i]=self.TD(value,state_batch,next_state_batch,reward_batch)
                    value_gradient=tape.gradient(self._TD[i],self.value_p)
                    actor_gradient=self._TD[i]*tape.gradient(tf.math.log(action_batch),self.actor_p)
                    if self.opt_flag==True:
                        self.optimizer(value_gradient,self.value_p)
                        self.optimizer(actor_gradient,self.actor_p)
                    else:
                        self.optimizer.apply_gradients(zip(value_gradient,self.value_p))
                        self.optimizer.apply_gradients(zip(actor_gradient,self.value_p))
                    self.loss[i]+=self._TD[i]
                    self.update_parameter()
            self.use_flag[i]=False
            if len(self.state_pool)<self.batch:
                self.loss[i]=self.loss[i].numpy()
            else:
                self.loss[i]=self.loss[i].numpy()/batches
        return
    
    
    def learn(self,episode_num,i):
        self.t+=1
        self.t_counter+=1
        self.one_list.append(1)
        self.index_list.append(i)
        self.use_flag.append(False)
        self.p=np.array(self.one_list,dtype=np.float16)/self.t_counter
        self.TD.append(0)
        self.loss.append(0)
        if len(self.state_pool)==i-1:
            self.state_pool.append(None)
            self.action_pool.append(None)
            self.next_state_pool.append(None)
            self.reward_pool.append(None)
        for _ in range(episode_num):
            episode=[]
            s=int(np.random.uniform(0,len(self.state_name)))
            if self.episode_step==None:
                while True:
                    next_s,end,_episode=self.explore(s,i)
                    s=next_s
                    self._learn(i)
                    if self.save_episode==True:
                        episode.append(_episode)
                    if end:
                        if self.save_episode==True:
                            self.episode.append(episode)
                        break
            else:
                for _ in range(self.episode_step):
                    next_s,end,_episode=self.explore(s,i)
                    s=next_s
                    self._learn(i)
                    if self.save_episode==True:
                        episode.append(_episode)
                    if end:
                        if self.save_episode==True:
                            self.episode.append(episode)
                        break
                if self.save_episode==True:
                    self.episode.append(episode)
        self.finish_list.append(i)
        self.t_counter-=1
        self.one_list[i]=0
        self.p=np.array(self.one_list,dtype=np.float16)/self.t_counter
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
        pickle.dump(self.actor_p,parameter_file)
        parameter_file.close()
        return
    
    
    def save_e(self,path):
        episode_file=open(path+'.dat','wb')
        pickle.dump(self.episode,episode_file)
        episode_file.close()
        return
    
    
    def save(self,path):
        output_file=open(path+'\save.dat','wb')
        path=path+'\save.dat'
        index=path.rfind('\\')
        if self.save_episode==True:
            episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
            pickle.dump(self.episode,episode_file)
            episode_file.close()
        self.episode_num=self.epi_num
        self.thread=self.t
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.tau,output_file)
        pickle.dump(self.end_loss,output_file)
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
    
    
    def restore(self,s_path,e_path=None):
        input_file=open(s_path,'rb')
        if self.save_episode==True:
            episode_file=open(e_path,'rb')
            self.episode=pickle.load(episode_file)
            episode_file.close()
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.tau=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
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
