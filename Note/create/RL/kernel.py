import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class kernel:
    def __init__(self,nn,update_param,optimizer,state,state_name,action_name,exploration_space,thread_lock,exploration=None,pr=None,pool_net=True,save_episode=True):
        self.nn=nn
        self._nn=nn.nn
        self.param=nn.param
        self._loss=nn.loss
        self.update_param=nn.update_param
        self.optimizer=nn.optimizer
        self.state_pool=[]
        self.action_pool=[]
        self.next_state_pool=[]
        self.reward_pool=[]
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.exploration=exploration
        self.pr=pr
        self.epsilon=[]
        self.discount=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.rp=None
        self.alpha=None
        self.beta=None
        self.lr=optimizer.lr
        self.end_loss=None
        self.thread=0
        self.thread_sum=0
        self.thread_lock=thread_lock
        self.one_list=[]
        self._one_list=[]
        self.use_flag=[]
        self.p=[]
        self.finish_list=[]
        self.pool_net=pool_net
        self.save_episode=save_episode
        self.TD=[]
        self.loss=[]
        self.loss_list=[]
        self.opt_flag==False
        self.a=[]
        self.epi_num=[]
        self.episode_num=[]
        self.total_episode=0
        self.time=0
        self.total_time=0
        
        
    def init(self,dtype=np.int32):
        t3=time.time()
        self.action_len=len(self.action_name)
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            if self.epsilon!=None:
                self.action_one=np.ones(len(self.action_name),dtype=dtype)
        t4=time.time()
        self.time+=t4-t3
        return
    
    
    def set_up(self,param=None,optimizer=None,discount=None,episode_num=None,episode_step=None,pool_size=None,batch=None,update_step=None,rp=None,alpha=None,beta=None,lr=None,end_loss=None,init=False):
        if param!=None:
            self.param=param
        if optimizer!=None:
            self.optimizer=optimizer
        if discount!=None:
            self.discount=discount
        if episode_num!=None:
            self.epi_num=episode_num
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
            self.index=np.arange(batch,dtype=np.int8)
        if update_step!=None:
            self.update_step=update_step
        if rp!=None:
            self.rp=rp
        if alpha!=None:
            self.alpha=alpha
        if beta!=None:
            self.beta=beta
        if lr!=None:
            self.lr=lr
        elif optimizer!=None:
            self.lr=optimizer.lr
        if end_loss!=None:
            self.end_loss=end_loss
        if init==True:
            self.thread=0
            self.thread_sum=0
            self.one_list=[]
            self._one_list=[]
            self.use_flag=[]
            self.p=[]
            self.finish_list=[]
            self.pool_net=True
            self.episode=[]
            self.epsilon=[]
            self.state_pool=None
            self.action_pool=None
            self.next_state_pool=None
            self.reward_pool=None
            self.TD=[]
            self.loss=[]
            self.loss_list=[]
            self.a=[]
            self.epi_num=[]
            self.episode_num=[]
            self.total_episode=0
            self.time=0
            self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one,epsilon):
        action_prob=action_one
        action_prob=action_prob*epsilon/len(action_one)
        best_a=np.argmax(self._nn(self.state[self.state_name[s]]))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def _epsilon_greedy_policy(self,a,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_a=np.argmax(a)
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def explore(self,s,epsilon,i):
        if type(self._nn)!=list:
            if self.exploration==None:
                action_prob=self.epsilon_greedy_policy(s,self.action_one,epsilon)
                a=np.random.choice(self.action,p=action_prob)
                next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
            else:
                if self.exploration_space==None:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration.explore(self.action_name[a])
                else:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.exploration.explore(self.state_name[s],self.action_name[a],self.exploration_space[self.state_name[s]][self.action_name[a]])
        else:
            if self.exploration==None:
                a=self._nn[1](self.state[self.state_name[s]],param=2).numpy()
                if len(a.shape)>0:
                    a=self._epsilon_greedy_policy(a,self.action_one)
                    next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                else:
                    next_s,r,end=self.exploration_space(self.state_name[s],a)
            else:
                if self.exploration_space==None:
                    a=self._nn[1](self.state[self.state_name[s]],param=2).numpy()
                    if len(a.shape)>0:
                        a=self._epsilon_greedy_policy(a,self.action_one)
                        next_s,r,end=self.exploration.explore(self.action_name[a])
                    else:
                        next_s,r,end=self.exploration.explore(a)
                else:
                    a=self._nn[1](self.state[self.state_name[s]],param=2).numpy()
                    if len(a.shape)>0:
                        a=self._epsilon_greedy_policy(a,self.action_one)
                        next_s,r,end=self.exploration.explore(self.state_name[s],self.action_name[a],self.exploration_space[self.state_name[s]][self.action_name[a]])
        if len(self._one_list)==i-1:
            self._one_list.append(self.one_list)
        else:
            self._one_list=self.one_list
        if len(self.p)==i-1:
            self.p.append(np.array(self._one_list[i],dtype=np.float16)/len(self._one_list))
        else:
            self.p[i]=np.array(self._one_list[i],dtype=np.float16)/len(self._one_list)
        if self.pool_net==True:
            flag=np.random.randint(0,2)
            while True:
                index=np.random.choice(len(self.p[i]),p=self.p[i])
                if index in self.finish_list or self.use_flag[i]==True:
                    continue
                else:
                    break
        if self.pool_net==True and flag==1 and self.state_pool[index]!=None and self.use_flag[i]==False:
            if self.exploration_space==None:
                self.state_pool[index]=tf.concat([self.state_pool[index],tf.expand_dims(s,axis=0)])
                self.action_pool[index]=tf.concat([self.action_pool[index],tf.expand_dims(a,axis=0)])
                self.next_state_pool[index]=tf.concat([self.next_state_pool[index],tf.expand_dims(next_s,axis=0)])
                self.reward_pool[index]=tf.concat([self.reward_pool[index],tf.expand_dims(r,axis=0)])
            else:
                self.state_pool[index]=tf.concat([self.state_pool[index],tf.expand_dims(self.state[self.state_name[s]],axis=0)])
                self.action_pool[index]=tf.concat([self.action_pool[index],tf.expand_dims(a,axis=0)])
                self.next_state_pool[index]=tf.concat([self.next_state_pool[index],tf.expand_dims(self.state[self.state_name[next_s]],axis=0)])
                self.reward_pool[index]=tf.concat([self.reward_pool[index],tf.expand_dims(r,axis=0)])
        else:
            if self.state_pool[i]==None:
                if self.exploration_space==None:
                    self.state_pool[i]=tf.expand_dims(s,axis=0)
                    self.action_pool[i]=tf.expand_dims(a,axis=0)
                    self.next_state_pool[i]=tf.expand_dims(next_s,axis=0)
                    self.reward_pool[i]=tf.expand_dims(r,axis=0)
                else:
                    self.state_pool[i]=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                    self.action_pool[i]=tf.expand_dims(a,axis=0)
                    self.next_state_pool[i]=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool[i]=tf.expand_dims(r,axis=0)
            else:
                if self.exploration_space==None:
                    self.state_pool[i]=tf.concat([self.state_pool[i],tf.expand_dims(s,axis=0)])
                    self.action_pool[i]=tf.concat([self.action_pool[i],tf.expand_dims(a,axis=0)])
                    self.next_state_pool[i]=tf.concat([self.next_state_pool[i],tf.expand_dims(next_s,axis=0)])
                    self.reward_pool[i]=tf.concat([self.reward_pool[i],tf.expand_dims(r,axis=0)])
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
                if self.exploration_space==None:
                    episode=[s,a,next_s,r,end]
                else:
                    if len(a.shape)>0:
                        episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r,end]
                    else:
                        episode=[self.state_name[s],a,self.state_name[next_s],r,end] 
        elif self.save_episode==True:
            if self.exploration_space==None:
                episode=[s,a,next_s,r]
            else:
                if len(a.shape)>0:
                    episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                else:
                    episode=[self.state_name[s],a,self.state_name[next_s],r]
        return next_s,end,episode,index
    
    
    def _learn(self,i):
        self.use_flag[i]=True
        if len(self.state_pool)<self.batch:
            with tf.GradientTape() as tape:
                if type(self._nn)!=list:
                    self.loss[i]=self._loss(self._nn,self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i])
                else:
                    value=self._nn[0](self.state_pool[i],param=0)
                    self.TD[i]=tf.reduce_mean((self.reward_pool+self.discount*self._nn[0](self.next_state_pool[i],param=1)-value)**2)
            if type(self._nn)!=list:
                gradient=tape.gradient(self.loss[i],self.param[0])
                self.optimizer.opt(gradient,self.param[0])
            else:
                value_gradient=tape.gradient(self.TD[i],self.param[0])
                actor_gradient=self.TD[i]*tape.gradient(tf.math.log(self.action_pool[i]),self.param[2])
                self.loss[i]=self.TD[i]
                self.optimizer.opt(value_gradient,actor_gradient,self.param)
            if self.update_step!=None:
                if self.a%self.update_step==0:
                    self.update_param.update(self.param)
            else:
                self.update_param.update(self.param)
        else:
            self.loss[i]=0
            batches=int((len(self.state_pool[i])-len(self.state_pool[i])%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if self.pool_net==True:
                for j in range(batches):
                    if self.pr!=None:
                        state_batch,action_batch,next_state_batch,reward_batch=self.pr(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.pool_size,self.batch,self.rp,self.alpha,self.beta)
                    else:
                        index1=j*self.batch
                        index2=(j+1)*self.batch
                        state_batch=self.state_pool[i][index1:index2]
                        action_batch=self.action_pool[i][index1:index2]
                        next_state_batch=self.next_state_pool[i][index1:index2]
                        reward_batch=self.reward_pool[i][index1:index2]
                    with tf.GradientTape() as tape:
                        if type(self._nn)!=list:
                            batch_loss=self._loss(self._nn,state_batch,action_batch,next_state_batch,reward_batch)
                        else:
                            value=self._nn[0](state_batch,param=0)
                            self.TD[i]=tf.reduce_mean((reward_batch+self.discount*self._nn[0](next_state_batch,param=1)-value)**2)
                    if type(self._nn)!=list:
                        gradient=tape.gradient(batch_loss,self.param[0])
                        self.optimizer.opt(gradient,self.param[0],self.lr)
                        self.loss[i]+=batch_loss
                    else:
                        value_gradient=tape.gradient(self.TD[i],self.param[0])
                        actor_gradient=self.TD[i]*tape.gradient(tf.math.log(action_batch),self.param[2])
                        self.optimizer.opt(value_gradient,actor_gradient,self.param)
                        self.loss[i]+=self.TD[i]
                if len(self.state_pool)%self.batch!=0:
                    if self.pr!=None:
                        state_batch,action_batch,next_state_batch,reward_batch=self.pr(self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i],self.pool_size,self.batch,self.rp,self.alpha,self.beta)
                    else:
                        batches+=1
                        index1=batches*self.batch
                        index2=self.batch-(self.shape0-batches*self.batch)
                        state_batch=tf.concat([self.state_pool[i][index1:],self.state_pool[i][:index2]])
                        action_batch=tf.concat([self.action_pool[i][index1:],self.action_pool[i][:index2]])
                        next_state_batch=tf.concat([self.next_state_pool[i][index1:],self.next_state_pool[i][:index2]])
                        reward_batch=tf.concat([self.reward_pool[i][index1:],self.reward_pool[i][:index2]])
                    with tf.GradientTape() as tape:
                        if type(self._nn)!=list:
                            batch_loss=self._loss(self._nn,state_batch,action_batch,next_state_batch,reward_batch)
                        else:
                            value=self._nn[0](state_batch,param=0)
                            self.TD[i]=tf.reduce_mean((reward_batch+self.discount*self._nn[0](next_state_batch,param=1)-value)**2)
                    if type(self._nn)!=list:
                        gradient=tape.gradient(batch_loss,self.param[0])
                        self.optimizer.opt(gradient,self.param[0],self.lr)
                        self.loss[i]+=batch_loss
                    else:
                        value_gradient=tape.gradient(self.TD[i],self.param[0])
                        actor_gradient=self.TD[i]*tape.gradient(tf.math.log(action_batch),self.param[2])
                        self.optimizer.opt(value_gradient,actor_gradient,self.param)
                        self.loss[i]+=self.TD[i]
            else:
                train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i])).shuffle(len(self.state_pool[i])).batch(self.batch)
                for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
                    with tf.GradientTape() as tape:
                        if type(self._nn)!=list:
                            batch_loss=self._loss(self._nn,state_batch,action_batch,next_state_batch,reward_batch)
                        else:
                            value=self._nn[0](state_batch,param=0)
                            self.TD[i]=tf.reduce_mean((reward_batch+self.discount*self._nn[0](next_state_batch,param=1)-value)**2)
                    if type(self._nn)!=list:
                        gradient=tape.gradient(batch_loss,self.param[0])
                        self.optimizer.opt(gradient,self.param[0],self.lr)
                        self.loss[i]+=batch_loss
                    else:
                        value_gradient=tape.gradient(self.TD[i],self.param[0])
                        actor_gradient=self.TD[i]*tape.gradient(tf.math.log(action_batch),self.param[2])
                        self.optimizer.opt(value_gradient,actor_gradient,self.param)
                        self.loss[i]+=self.TD[i]
            self.use_flag[i]=False
            if self.update_step!=None:
                if self.a%self.update_step==0:
                    self.update_param.update(self.param)
            else:
                self.update_param.update(self.param)
            if len(self.state_pool)<self.batch:
                self.loss[i]=self.loss[i].numpy()
            else:
                self.loss[i]=self.loss[i].numpy()/batches
        return
    
    
    def learn(self,epsilon,episode_num,i):
        self.thread_lock.acquire()
        self.thread+=1
        self.thread_lock.release()
        if len(self.state_pool)==i:
            self.state_pool.append(None)
            self.action_pool.append(None)
            self.next_state_pool.append(None)
            self.reward_pool.append(None)
            self.epsilon.append(epsilon)
            self.epi_num.append(episode_num)
            self.episode_num.append(0)
            self.one_list.append(1)
            self.use_flag.append(False)
            self.thread_lock.acquire()
            self.thread_sum+=1
            self.thread_lock.release()
        elif i not in self.finish_list:
            self.one_list[i]=1
        self.a.append(0)
        self.TD.append(0)
        self.loss.append(0)
        for _ in range(episode_num):
            if self.episode_num[i]==self.epi_num[i]:
                break
            self.episode_num[i]+=1
            episode=[]
            if self.exploration_space==None:
                s=self.exploration.explore(init=True)
            else:
                s=int(np.random.uniform(0,len(self.state_name)))
            if self.episode_step==None:
                while True:
                    self.a[i]+=1
                    next_s,end,_episode,index=self.explore(s,self.epsilon[i],i)
                    s=next_s
                    self._learn(i)
                    if self.save_episode==True:
                        if index not in self.finish_list:
                            episode.append(_episode)
                    if end:
                        if self.save_episode==True:
                            self.episode.append(episode)
                        break
            else:
                for _ in range(self.episode_step):
                    self.a[i]+=1
                    next_s,end,episode=self.explore(s,self.epsilon[i],i)
                    s=next_s
                    self._learn(i)
                    if self.save_episode==True:
                        if index not in self.finish_list:
                            episode.append(_episode)
                    if end:
                        if self.save_episode==True:
                            self.episode.append(episode)
                        break
                if self.save_episode==True:
                    self.episode.append(episode)
        if i not in self.finish_list:
            self.finish_list.append(i)
        self.thread_lock.acquire()
        self.thread-=1
        self.thread_lock.release()
        self.one_list[i]=0
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
        parameter_file=open(path.replace(path[index+1:],'parameter.dat'),'wb')
        if self.save_episode==True:
            episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
            pickle.dump(self.episode,episode_file)
            episode_file.close()
        self.one_list=[0]*len(self.one_list)
        self.use_flag=[False]*len(self.use_flag)
        pickle.dump(self.param,parameter_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.exploration,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.index,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_param,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.rp,output_file)
        pickle.dump(self.alpha,output_file)
        pickle.dump(self.beta,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.thread_sum,output_file)
        pickle.dump(self.one_list,output_file)
        pickle.dump(self._one_list,output_file)
        pickle.dump(self.use_flag,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.finish_list,output_file)
        pickle.dump(self.pool_net,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.opt_flag,output_file)
        pickle.dump(self.epi_num,output_file)
        pickle.dump(self.episode_num,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,p_path,e_path=None):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        if e_path!=None:
            episode_file=open(e_path,'rb')
            self.episode=pickle.load(episode_file)
            episode_file.close()
        self.nn.param=pickle.load(parameter_file)
        self.param=self.nn.param
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.exploration=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.index=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_param=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.rp=pickle.load(input_file)
        self.alpha=pickle.load(input_file)
        self.beta=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.thread_sum=pickle.load(input_file)
        self.one_list=pickle.load(input_file)
        self._one_list=pickle.load(input_file)
        self.use_flag=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.finish_list=pickle.load(input_file)
        self.pool_net=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.opt_flag=pickle.load(input_file)
        self.epi_num=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
