from tensorflow import data as tf_data
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import os
import time


class kernel:
    def __init__(self,nn=None,state=None,state_name=None,action_name=None,save_episode=True):
        if nn!=None:
            self.nn=nn
            try:
                self.nn.km=1
            except AttributeError:
                pass
        self.ol=None
        self.PO=None
        self.thread_lock=None
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.action_num=None
        self.epsilon=None
        self.discount=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.trial_num=None
        self.criterion=None
        self.reward_list=[]
        self.suspend=False
        self.stop=None
        self.stop_flag=1
        self.save_epi=None
        self.action_prob=0
        self.train_counter=0
        self.end_loss=None
        self.save_episode=save_episode
        self.loss=None
        self.loss_list=[]
        self.step_counter=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def action_init(self,action_num=None,dtype=np.int32):
        if self.action_name!=None:
            self.action_num=len(self.action_name)
        else:
            action_num=self.action_num
            self.action_num=action_num
        if self.action_name!=None and len(self.action_name)>self.action_num:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_num,dtype=dtype)+self.action_num))
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_num,dtype=dtype)))
        elif self.action_name!=None:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            if self.epsilon!=None:
                self.action_one=np.ones(len(self.action_name),dtype=dtype)
        if self.action_num>action_num:
            self.action=np.concatenate((self.action,np.arange(self.action_num-action_num,dtype=dtype)+action_num))
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(self.action_num-action_num,dtype=dtype)))
        else:
            self.action=np.arange(self.action_num,dtype=dtype)
            if self.epsilon!=None:
                self.action_one=np.ones(self.action_num,dtype=dtype)
        return
    
    
    def set_up(self,epsilon=None,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_num=None,criterion=None,end_loss=None):
        if epsilon!=None:
            self.epsilon=epsilon
        if discount!=None:
            self.discount=discount
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_step!=None:
            self.update_step=update_step
        if trial_num!=None:
            self.trial_num=trial_num
        if criterion!=None:
            self.criterion=criterion
        if end_loss!=None:
            self.end_loss=end_loss
        self.episode=[]
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.reward_list=[]
        self.loss=0
        self.loss_list=[]
        self.step_counter=0
        self.total_episode=0
        self.time=0
        self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one):
        action_prob=action_one*self.epsilon/len(action_one)
        if self.state==None:
            best_a=np.argmax(self.nn.nn(s,target=False))
        else:
            best_a=np.argmax(self.nn.nn(self.state[self.state_name[s]]))
        action_prob[best_a.numpy()]+=1-self.epsilon
        return action_prob
    
    
    def get_episode(self,s):
        next_s=None
        episode=[]
        self.end_flag=False
        while True:
            s=next_s
            try:
                if self.nn.nn!=None:
                    pass
                try:
                    if self.nn.explore!=None:
                        pass
                    s=np.expand_dims(s,axis=0)
                    a=np.argmax(self.nn.nn(s,target=False)).numpy()
                    if self.action_name==None:
                        next_s,r,done=self.nn.explore(a)
                    else:
                        next_s,r,done=self.nn.explore(self.action_name[a])
                except AttributeError:
                    a=np.argmax(self.nn.nn(s,target=False))
                    next_s,r,done=self.nn.transition(self.state_name[s],self.action_name[a])
            except AttributeError:
                try:
                    if self.nn.explore!=None:
                        pass
                    if self.state_name==None:
                        s=np.expand_dims(s,axis=0)
                        a=self.nn.actor(s,target=False).numpy()
                    else:
                        a=self.nn.actor(self.state[self.state_name[s]],target=False).numpy()
                    next_s,r,done=self.nn.explore(a)
                except AttributeError:
                    a=self.nn.actor(self.state[self.state_name[s]],target=False).numpy()
                    next_s,r,done=self.nn.transition(self.state_name[s],a)
            if done:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append([s,self.action_name[a],next_s,r])
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
                episode.append('done')
                break
            elif self.end_flag==True:
                break
            else:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append([s,self.action_name[a],next_s,r])
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
        return episode
    
    
    def end(self):
        if self.end_loss!=None and self.loss_list[-1]<=self.end_loss:
            return True
    
    
    def opt(self,state_batch=None,action_batch=None,next_state_batch=None,reward_batch=None):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch)
        self.nn.opt(loss)
        return loss
    
    
    def opt_t(self,data):
        loss=self.nn.loss(data)
        if self.thread_lock!=None:
            if self.PO==1:
                self.thread_lock[0].acquire()
                self.nn.opt(loss)
                self.thread_lock[0].release()
        else:
            self.nn.opt(loss)
        return loss.numpy()
    
    
    def pool(self,s,a,next_s,r):
        if type(self.state_pool)!=np.ndarray and self.state_pool==None:
            if self.state==None:
                self.state_pool=s
                self.action_pool=np.expand_dims(a,axis=0)
                self.next_state_pool=np.expand_dims(next_s,axis=0)
                self.reward_pool=np.expand_dims(r,axis=0)
            else:
                self.state_pool=np.expand_dims(self.state[self.state_name[s]],axis=0)
                self.action_pool=np.expand_dims(a,axis=0)
                self.next_state_pool=np.expand_dims(self.state[self.state_name[next_s]],axis=0)
                self.reward_pool=np.expand_dims(r,axis=0)
        else:
            if self.state==None:
                self.state_pool=np.concatenate((self.state_pool,s),0)
                self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0)
                self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(next_s,axis=0)),0)
                self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0)
            else:
                self.state_pool=np.concatenate((self.state_pool,np.expand_dims(self.state[self.state_name[s]],axis=0)),0)
                self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0)
                self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(self.state[self.state_name[next_s]],axis=0)),0)
                self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0)
        if len(self.state_pool)>self.pool_size:
            self.state_pool=self.state_pool[1:]
            self.action_pool=self.action_pool[1:]
            self.next_state_pool=self.next_state_pool[1:]
            self.reward_pool=self.reward_pool[1:]
        return
    
    
    def _train(self,state=None,action=None,next_state=None,reward=None):
        if len(self.state_pool)<self.batch:
            return
        else:
            loss=0
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            try:
                if self.nn.data_func!=None:
                    pass
                for j in range(batches):
                    self.suspend_func()
                    state_batch,action_batch,next_state_batch,reward_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch)
                    loss+=batch_loss
                    try:
                        self.nn.bc=j
                    except AttributeError:
                        pass
                if len(self.state_pool)%self.batch!=0:
                    self.suspend_func()
                    state_batch,action_batch,next_state_batch,reward_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch)
                    loss+=batch_loss
                    try:
                        self.nn.bc+=1
                    except AttributeError:
                        pass
            except AttributeError:
                j=0
                train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                try:
                    self.nn.bc=0
                except AttributeError:
                    pass
                for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch)
                    loss+=batch_loss
                    j+=1
                    try:
                        self.nn.bc+=1
                    except AttributeError:
                        pass
            if self.update_step!=None:
                if self.step_counter%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
        loss=loss.numpy()/batches
        return loss
    
    
    def train_(self):
        episode=[]
        self.reward=0
        if self.state_name==None:
            s=self.nn.explore(init=True)
        else:
            s=int(np.random.uniform(0,len(self.state_name)))
        if self.episode_step==None:
            while True:
                t1=time.time()
                try:
                    if self.nn.nn!=None:
                        pass
                    try:
                        if self.nn.explore!=None:
                            pass
                        s=np.expand_dims(s,axis=0)
                        if self.epsilon==None:
                            self.epsilon=self.nn.epsilon(self.step_counter)
                        try:
                            if self.nn.action!=None:
                                pass
                            a=self.nn.action(s)
                        except AttributeError:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                        if self.action_name==None:
                            next_s,r,done=self.nn.explore(a)
                        else:
                            next_s,r,done=self.nn.explore(self.action_name[a])
                    except AttributeError:
                        if self.epsilon==None:
                            self.epsilon=self.nn.epsilon(self.step_counter)
                        try:
                            if self.nn.action!=None:
                                pass
                            a=self.nn.action(s)
                        except AttributeError:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                        next_s,r,done=self.nn.transition(self.state_name[s],self.action_name[a])
                    self.pool(s,a,next_s,r)
                except AttributeError:
                    try:
                        if self.nn.explore!=None:
                            pass 
                        if self.state_name==None:
                            s=np.expand_dims(s,axis=0)
                            a=(self.nn.actor(s,target=False)+np.random.normal([1])).numpy()
                        else:
                            a=(self.nn.actor(self.state[self.state_name[s]],target=False)+np.random.normal([1])).numpy()
                        next_s,r,done=self.nn.explore(a)
                    except AttributeError:
                        a=(self.nn.actor(self.state[self.state_name[s]],target=False)+np.random.normal([1])).numpy()
                        next_s,r,done=self.nn.transition(self.state_name[s],a)
                    self.pool(s,a,next_s,r)
                self.reward=r+self.reward
                loss=self._train()
                self.step_counter+=1
                if done:
                    if self.save_episode==True:
                        if self.state_name==None and self.action_name==None:
                            episode=[s,a,next_s,r]
                        elif self.state_name==None:
                            episode=[s,self.action_name[a],next_s,r]
                        elif self.action_name==None:
                            episode=[self.state_name[s],a,self.state_name[next_s],r]
                        else:
                            episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                    self.reward_list.append(self.reward)
                    t2=time.time()
                    self.time+=(t2-t1)
                    return loss,episode,done
                elif self.save_episode==True:
                    if self.state_name==None and self.action_name==None:
                        episode=[s,a,next_s,r]
                    elif self.state_name==None:
                        episode=[s,self.action_name[a],next_s,r]
                    elif self.action_name==None:
                        episode=[self.state_name[s],a,self.state_name[next_s],r]
                    else:
                        episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                s=next_s
        else:
            for _ in range(self.episode_step):
                t1=time.time()
                try:
                    if self.nn.nn!=None:
                        pass
                    try:
                        if self.nn.explore!=None:
                            pass
                        s=np.expand_dims(s,axis=0)
                        if self.epsilon==None:
                            self.epsilon=self.nn.epsilon(self.step_counter)
                        try:
                            if self.nn.action!=None:
                                pass
                            a=self.nn.action(s)
                        except AttributeError:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                        if self.action_name==None:
                            next_s,r,done=self.nn.explore(a)
                        else:
                            next_s,r,done=self.nn.explore(self.action_name[a])
                    except AttributeError:
                        if self.epsilon==None:
                            self.epsilon=self.nn.epsilon(self.step_counter)
                        try:
                            if self.nn.action!=None:
                                pass
                            a=self.nn.action(s)
                        except AttributeError:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                        next_s,r,done=self.nn.transition(self.state_name[s],self.action_name[a])
                    self.pool(s,a,next_s,r)
                except AttributeError:
                    try:
                        if self.nn.explore!=None:
                            pass 
                        if self.state_name==None:
                            s=np.expand_dims(s,axis=0)
                            a=(self.nn.actor(s,target=False)+np.random.normal([1])).numpy()
                        else:
                            a=(self.nn.actor(self.state[self.state_name[s]],target=False)+np.random.normal([1])).numpy()
                        next_s,r,done=self.nn.explore(a)
                    except AttributeError:
                        a=(self.nn.actor(self.state[self.state_name[s]],target=False)+np.random.normal([1])).numpy()
                        next_s,r,done=self.nn.transition(self.state_name[s],a)
                    self.pool(s,a,next_s,r)
                self.reward=r+self.reward
                loss=self._train()
                self.step_counter+=1
                if done:
                    if self.save_episode==True:
                        if self.state_name==None and self.action_name==None:
                            episode=[s,a,next_s,r]
                        elif self.state_name==None:
                            episode=[s,self.action_name[a],next_s,r]
                        elif self.action_name==None:
                            episode=[self.state_name[s],a,self.state_name[next_s],r]
                        else:
                            episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                    self.reward_list.append(self.reward)
                    t2=time.time()
                    self.time+=(t2-t1)
                    return loss,episode,done
                elif self.save_episode==True:
                    if self.state_name==None and self.action_name==None:
                        episode=[s,a,next_s,r]
                    elif self.state_name==None:
                        episode=[s,self.action_name[a],next_s,r]
                    elif self.action_name==None:
                        episode=[self.state_name[s],a,self.state_name[next_s],r]
                    else:
                        episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                s=next_s
        self.reward_list.append(self.reward)
        t2=time.time()
        self.time+=(t2-t1)
        return loss,episode,done
    
    
    def train(self,episode_num,save=None,one=True,p=None,s=None):
        self.train_counter+=1
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if s==None:
            self.s=1
            self.file_list=None
        else:
            self.s=s-1
            self.file_list=[]
        if episode_num!=None:
            for i in range(episode_num):
                if self.stop==True:
                    if self.stop_func():
                        return
                loss,episode,done=self.train_()
                if self.trial_num!=None:
                    if len(self.reward_list)>=self.trial_num:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_num:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            self._time=self.time-int(self.time)
                            if self._time<0.5:
                                self.time=int(self.time)
                            else:
                                self.time=int(self.time)+1
                            self.total_time+=self.time
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.6f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                self.loss=loss
                self.loss_list.append(loss)
                self.epi_num+=1
                self.total_episode+=1
                if episode_num%10!=0:
                    p=episode_num-episode_num%self.p
                    p=int(p/self.p)
                    s=episode_num-episode_num%self.s
                    s=int(s/self.s)
                else:
                    p=episode_num/(self.p+1)
                    p=int(p)
                    s=episode_num/(self.s+1)
                    s=int(s)
                if p==0:
                    p=1
                if s==0:
                    s=1
                if i%p==0:
                    if self.train_counter==1:
                        if self.state_pool!=None and len(self.state_pool)>=self.batch:
                            print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                        else:
                            print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                        if avg_reward!=None:
                            print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                        else:
                            print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                        print()
                    else:
                        if self.state_pool!=None and len(self.state_pool)>=self.batch:
                            print('episode:{0}   loss:{1:.6f}'.format(self.total_episode,loss))
                        else:
                            print('episode:{0}   loss:{1:.6f}'.format(self.total_episode,loss))
                        if avg_reward!=None:
                            print('episode:{0}   average reward:{1}'.format(self.total_episode,avg_reward))
                        else:
                            print('episode:{0}   reward:{1}'.format(self.total_episode,self.reward))
                        print()
                if save!=None and i%s==0:
                    self.save(self.total_episode,one)
                if self.save_episode==True:
                    if done:
                        episode.append('done')
                    self.episode.append(episode)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
        elif self.ol==None:
            i=0
            while True:
                if self.stop==True:
                    if self.stop_func():
                        return
                loss,episode,done=self.train_()
                if self.trial_num!=None:
                    if len(self.reward_list)==self.trial_num:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_num:])
                        if avg_reward>=self.criterion:
                            self._time=self.time-int(self.time)
                            if self._time<0.5:
                                self.time=int(self.time)
                            else:
                                self.time=int(self.time)+1
                            self.total_time+=self.time
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.6f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            self.train_flag=False
                            return
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.epi_num+=1
                self.total_episode+=1
                if episode_num%10!=0:
                    p=episode_num-episode_num%self.p
                    p=int(p/self.p)
                    s=episode_num-episode_num%self.s
                    s=int(s/self.s)
                else:
                    p=episode_num/(self.p+1)
                    p=int(p)
                    s=episode_num/(self.s+1)
                    s=int(s)
                if p==0:
                    p=1
                if s==0:
                    s=1
                if i%p==0:
                    if self.train_counter==1:
                        if len(self.state_pool)>=self.batch:
                            print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                        if avg_reward!=None:
                            print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                        else:
                            print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                        print()
                    else:
                        if len(self.state_pool)>=self.batch:
                            print('episode:{0}   loss:{1:.6f}'.format(self.total_episode,loss))
                        if avg_reward!=None:
                            print('episode:{0}   average reward:{1}'.format(self.total_episode,avg_reward))
                        else:
                            print('episode:{0}   reward:{1}'.format(self.total_episode,self.reward))
                        print()
                if save!=None and i%s==0:
                    self.save(self.total_episode,one)
                if self.save_episode==True:
                    if done:
                        episode.append('done')
                    self.episode.append(episode)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
        else:
            data=self.ol()
            loss=self.opt_t(data)
            if self.thread_lock!=None:
                if self.PO==1:
                    self.thread_lock[1].acquire()
                self.loss=loss
                self.nn.train_loss.append(loss.astype(np.float32))
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                self.total_episode+=1
                if self.PO==1:
                    self.thread_lock[1].release()
            else:
                self.loss=loss
                self.nn.train_loss.append(loss.astype(np.float32))
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                self.total_episode+=1
        if save!=None:
            self.save()
        self._time=self.time-int(self.time)
        if self._time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print('last loss:{0:.6f}'.format(loss))
        print('last reward:{0}'.format(self.reward))
        print()
        print('time:{0}s'.format(self.total_time))
        return
        
    
    def suspend_func(self):
        if self.suspend==True:
            if self.save_epoch==None:
                print('Training have suspended.')
            else:
                self._save()
            while True:
                if self.suspend==False:
                    print('Training have continued.')
                    break
        return
    
    
    def stop_func(self):
        if self.end():
            self.save(self.total_episode,True)
            print('\nSystem have stopped training,Neural network have been saved.')
            self._time=self.time-int(self.time)
            if self._time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            self.total_time+=self.time
            print('episode:{0}'.format(self.total_episode))
            print('last loss:{0:.6f}'.format(self.loss))
            print('reward:{0}'.format(self.reward))
            print()
            print('time:{0}s'.format(self.total_time))
            return True
        elif self.stop_flag==1:
            print('\nSystem have stopped training.')
            self._time=self.time-int(self.time)
            if self._time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            self.total_time+=self.time
            print('episode:{0}'.format(self.total_episode))
            print('last loss:{0:.6f}'.format(self.loss))
            print('reward:{0}'.format(self.reward))
            print()
            print('time:{0}s'.format(self.total_time))
            return True
        return False
    
        
    def _save(self):
        if self.save_epi==self.total_episode:
            self.save(self.total_episode,False)
            self.save_epi=None
            print('\nNeural network have saved and training have suspended.')
            return
        elif self.save_epi!=None and self.save_epi>self.total_episode:
            print('\nsave_epoch>total_epoch')
        return
    
    
    def reward_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_episode),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
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
    
    
    def save_e(self):
        episode_file=open('episode.dat','wb')
        pickle.dump(self.episode,episode_file)
        episode_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if one==True:
            output_file=open('save.dat','wb')
            if self.save_episode==True:
                episode_file=open('episode.dat','wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
        else:
            output_file=open('save-{0}.dat'.format(i),'wb')
            if self.save_episode==True:
                episode_file=open('episode-{0}.dat'.format(i),'wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
            if self.save_episode==True:
                self.file_list.append(['save-{0}.dat','param-{0}.dat','episode-{0}.dat'])
            else:
                self.file_list.append(['save-{0}.dat','param-{0}.dat'])
            if len(self.file_list)>self.s+1:
                os.remove(self.file_list[0][0])
                os.remove(self.file_list[0][1])
                os.remove(self.file_list[0][2])
                del self.file_list[0]
        pickle.dump(self.nn,output_file)
        pickle.dump(self.ol,output_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.action_num,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.train_counter,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.loss,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.step_counter,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.s,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,e_path=None):
        input_file=open(s_path,'rb')
        if e_path!=None:
            episode_file=open(e_path,'rb')
            self.episode=pickle.load(episode_file)
            episode_file.close()
        self.nn=pickle.load(input_file)
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.ol=pickle.load(input_file)
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.action_num=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.train_counter=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.step_counter=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.s=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
