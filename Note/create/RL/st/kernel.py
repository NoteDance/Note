from tensorflow import function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time


class kernel:
    def __init__(self,nn=None,state=None,state_name=None,action_name=None,save_episode=True):
        if nn!=None:
            self.nn=nn
            self.opt=nn.opt
            try:
                self.nn.km=1
            except AttributeError:
                pass
        self.core=None
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
        self.epsilon=None
        self.discount=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.suspend=False
        self.stop=None
        self.train_flag=None
        self.save_epi=None
        self.train_counter=0
        self.end_loss=None
        self.save_episode=save_episode
        self.loss_list=[]
        self.a=0
        self.d=None
        self.e=None
        self.epi_num=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def init(self,dtype=np.int32):
        self.action_len=len(self.action_name)
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            if self.epsilon!=None:
                self.action_one=np.ones(len(self.action_name),dtype=dtype)
        return
    
    
    def set_up(self,param=None,epsilon=None,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,end_loss=None):
        if param!=None:
            self.nn.param=param
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
        if end_loss!=None:
            self.end_loss=end_loss
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
        self.total_e=0
        self.time=0
        self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        if self.state==None:
            best_a=np.argmax(self.nn.nn(s))
        else:
            best_a=np.argmax(self.nn.nn(self.state[self.state_name[s]]))
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def _epsilon_greedy_policy(self,a,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_a=np.argmax(a)
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def get_episode(self,s):
        next_s=None
        episode=[]
        self.end=False
        while True:
            s=next_s
            if type(self.nn.nn)!=list:
                try:
                    if self.nn.explore!=None:
                        pass
                    a=np.argmax(self.nn.nn(s))
                    if self.action_name==None:
                        next_s,r,end=self.nn.explore(a)
                    else:
                        next_s,r,end=self.nn.explore(self.action_name[a])
                except AttributeError:
                    a=np.argmax(self.nn.nn(s))
                    next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
            else:
                try:
                    if self.nn.explore!=None:
                        pass
                    if len(self.nn.param)==4:
                        if self.state_name==None:
                            a=(self.nn.nn[1](s,p=1)+self.core.random.normal([1])).numpy()
                        else:
                            a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+self.core.random.normal([1])).numpy()
                    else:
                        if self.state_name==None:
                            a=self.nn.nn[1](s).numpy()
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                    if len(a.shape)>0:
                        a=np.argmax(a)
                        next_s,r,end=self.nn.explore(self.action_name[a])
                    else:
                        next_s,r,end=self.nn.explore(a)
                except AttributeError:
                    if len(self.nn.param)==4:
                        a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+self.core.random.normal([1])).numpy()
                    else:
                        a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                    if len(a.shape)>0:
                        a=np.argmax(a)
                        next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
                    else:
                        next_s,r,end=self.nn.transition(self.state_name[s],a)
            if end:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append([s,self.action_name[a],next_s,r])
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
                episode.append('end')
                break
            elif self.end==True:
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
    
    
    @function
    def tf_opt(self,state_batch=None,action_batch=None,next_state_batch=None,reward_batch=None):
        if len(self.state_pool)<self.batch:
            state_batch=self.state_pool 
            action_batch=self.action_pool
            next_state_batch=self.next_state_pool
            reward_batch=self.reward_pool  
        with self.core.GradientTape() as tape:
            if type(self.nn.nn)!=list:
                loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
            elif len(self.nn.param)==4:
                value=self.nn.nn[0](self.state_pool,p=0)
                TD=self.core.reduce_mean((self.reward_pool+self.discount*self.nn.nn[0](self.next_state_pool,p=2)-value)**2)
            else:
                value=self.nn.nn[0](state_batch)
                TD=self.core.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)  
        if type(self.nn.nn)!=list:
            gradient=tape.gradient(loss,self.nn.param[0])
            self.opt(gradient,self.nn.param[0])
            return loss
        elif len(self.nn.param)==4:
            value_gradient=tape.gradient(TD,self.nn.param[0])
            actor_gradient=tape.gradient(value,self.action_pool)*tape.gradient(self.action_pool,self.nn.param[1])
            self.opt(value_gradient,actor_gradient,self.nn.param)
            return TD
        else:
            value_gradient=tape.gradient(TD,self.nn.param[0])
            actor_gradient=TD*tape.gradient(self.core.math.log(action_batch),self.nn.param[1])
            self.opt(value_gradient,actor_gradient,self.nn.param)
            return TD
    
    
    @function
    def tf_opt_t(self,data):
        with self.core.GradientTape() as tape:
            if type(self.nn.nn)!=list:
                loss=self.nn.loss(self.nn.nn,data[0],data[1],data[2],data[3])				
            elif len(self.nn.param)==4:
                value=self.nn.nn[0](data[0],p=0)
                TD=self.core.reduce_mean((data[3]+self.discount*self.nn.nn[0](data[2],p=2)-value)**2)
            else:  
                value=self.nn.nn[0](data[0])
                TD=self.core.reduce_mean((data[3]+self.discount*self.nn.nn[0](data[2])-value)**2)
        if self.thread_lock!=None:
            if self.PO==1:
                if type(self.nn.nn)!=list:
                    gradient=tape.gradient(loss,self.nn.param[0])
                    self.opt(gradient,self.nn.param[0])
                elif len(self.nn.param)==4:
                    value_gradient=tape.gradient(TD,self.nn.param[0])				
                    actor_gradient=tape.gradient(value,data[1])*tape.gradient(data[1],self.nn.param[1])
                    loss=TD
                    self.opt(value_gradient,actor_gradient,self.nn.param)
                else:
                    value_gradient=tape.gradient(TD,self.nn.param[0])				
                    actor_gradient=TD*tape.gradient(self.core.math.log(data[1]),self.nn.param[1])
                    loss=TD
                    self.opt(value_gradient,actor_gradient,self.nn.param)
            else:
                if type(self.nn.nn)!=list:
                    self.thread_lock.acquire()
                    self.param=self.nn.param
                    self.gradient=tape.gradient(loss,self.param[0])
                    self.thread_lock.release()
                    self.thread_lock.acquire()
                    self.opt(self.gradient,self.nn.param[0])
                    self.thread_lock.release()
                elif len(self.nn.param)==4:
                    self.thread_lock.acquire()
                    self.param=self.nn.param
                    self.value_gradient=tape.gradient(TD,self.param[0])				
                    self.actor_gradient=tape.gradient(value,data[1])*tape.gradient(data[1],self.param[1])
                    self.thread_lock.release()
                    loss=TD
                    self.thread_lock.acquire()
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                    self.thread_lock.release()
                else:
                    self.thread_lock.acquire()
                    self.param=self.nn.param
                    self.value_gradient=tape.gradient(TD,self.param[0])				
                    self.actor_gradient=TD*tape.gradient(self.core.math.log(data[1]),self.param[1])
                    self.thread_lock.release()
                    loss=TD
                    self.thread_lock.acquire()
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                    self.thread_lock.release()
        else:
            if type(self.nn.nn)!=list:
                gradient=tape.gradient(loss,self.nn.param[0])
                self.opt(gradient,self.nn.param[0])
            elif len(self.nn.param)==4:
                value_gradient=tape.gradient(TD,self.nn.param[0])				
                actor_gradient=tape.gradient(value,data[1])*tape.gradient(data[1],self.nn.param[1])
                loss=TD
                self.opt(value_gradient,actor_gradient,self.nn.param)
            else:
                value_gradient=tape.gradient(TD,self.nn.param[0])				
                actor_gradient=TD*tape.gradient(self.core.math.log(data[1]),self.nn.param[1])
                loss=TD
                self.opt(value_gradient,actor_gradient,self.nn.param)
        return loss
    
    
    def opt(self,state_batch=None,action_batch=None,next_state_batch=None,reward_batch=None):
        try:
            if self.core.DType!=None:
                pass
            loss,TD=self.opt(state_batch,action_batch,next_state_batch,reward_batch)
            if type(self.nn.nn)!=list:
                return loss
            elif len(self.nn.param)==4:
                return TD
            else:
                return TD
        except AttributeError:
            pass
    
    
    def opt_t(self,data=None):
        try:
            if self.core.DType!=None:
                pass
            loss=self.tf_opt_t(data)
        except AttributeError:
            pass
        return loss
    
    
    def _train(self):
        if self.end_loss!=None:
            self.param=self.nn.param
        if len(self.state_pool)<self.batch:
            _=self.opt()
            if self.update_step!=None:
                if self.a%self.update_step==0:
                    self.nn.update_param(self.nn.param)
            else:
                self.nn.update_param(self.nn.param)
            loss=0
        else:
            loss=0
            self.loss=0
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
                train_ds=self.core.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool)).shuffle(len(self.state_pool)).batch(self.batch)
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
                if self.a%self.update_step==0:
                    self.nn.update_param(self.nn.param)
            else:
                self.nn.update_param(self.nn.param)
            if len(self.state_pool)<self.batch:
                loss=loss.numpy()
            else:
                loss=loss.numpy()/batches
        return loss
    
    
    def train_(self):
        episode=[]
        if self.state_name==None:
            s=self.nn.explore(init=True)
        else:
            s=int(np.random.uniform(0,len(self.state_name)))
        if self.episode_step==None:
            while True:
                t1=time.time()
                self.a+=1
                if type(self.nn.nn)!=list:
                    try:
                        if self.nn.explore!=None:
                            pass
                        action_prob=self.epsilon_greedy_policy(s,self.action_one)
                        a=np.random.choice(self.action,p=action_prob)
                        if self.action_name==None:
                            next_s,r,end=self.nn.explore(a)
                        else:
                            next_s,r,end=self.nn.explore(self.action_name[a])
                    except AttributeError:
                        action_prob=self.epsilon_greedy_policy(s,self.action_one)
                        a=np.random.choice(self.action,p=action_prob)
                        next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
                else:
                    try:
                        if self.nn.explore!=None:
                            pass 
                        if len(self.nn.param)==4:
                            if self.state_name==None:
                                a=(self.nn.nn[1](s,p=1)+self.core.random.normal([1])).numpy()
                            else:
                                a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+self.core.random.normal([1])).numpy()
                        else:
                            if self.state_name==None:
                                a=self.nn.nn[1](s).numpy()
                            else:
                                a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                        if len(a.shape)>0:
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.nn.explore(self.action_name[a])
                        else:
                            next_s,r,end=self.nn.explore(a)
                    except AttributeError:
                        if len(self.nn.param)==4:
                            a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+self.core.random.normal([1])).numpy()
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                        if len(a.shape)>0:
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
                        else:
                            next_s,r,end=self.nn.transition(self.state_name[s],a)
                if self.state_pool==None:
                    if self.state==None:
                        self.state_pool=self.core.expand_dims(s,axis=0)
                        self.action_pool=self.core.expand_dims(a,axis=0)
                        self.next_state_pool=self.core.expand_dims(next_s,axis=0)
                        self.reward_pool=self.core.expand_dims(r,axis=0)
                    else:
                        self.state_pool=self.core.expand_dims(self.state[self.state_name[s]],axis=0)
                        self.action_pool=self.core.expand_dims(a,axis=0)
                        self.next_state_pool=self.core.expand_dims(self.state[self.state_name[next_s]],axis=0)
                        self.reward_pool=self.core.expand_dims(r,axis=0)
                else:
                    if self.state==None:
                        self.state_pool=self.core.concat([self.state_pool,self.core.expand_dims(s,axis=0)],0)
                        self.action_pool=self.core.concat([self.action_pool,self.core.expand_dims(a,axis=0)],0)
                        self.next_state_pool=self.core.concat([self.next_state_pool,self.core.expand_dims(next_s,axis=0)],0)
                        self.reward_pool=self.core.concat([self.reward_pool,self.core.expand_dims(r,axis=0)],0)
                    else:
                        self.state_pool=self.core.concat([self.state_pool,self.core.expand_dims(self.state[self.state_name[s]],axis=0)],0)
                        self.action_pool=self.core.concat([self.action_pool,self.core.expand_dims(a,axis=0)],0)
                        self.next_state_pool=self.core.concat([self.next_state_pool,self.core.expand_dims(self.state[self.state_name[next_s]],axis=0)],0)
                        self.reward_pool=self.core.concat([self.reward_pool,self.core.expand_dims(r,axis=0)],0)
                if len(self.state_pool)>self.pool_size:
                    self.state_pool=self.state_pool[1:]
                    self.action_pool=self.action_pool[1:]
                    self.next_state_pool=self.next_state_pool[1:]
                    self.reward_pool=self.reward_pool[1:]
                if end:
                    if self.save_episode==True:
                        if self.state_name==None and self.action_name==None:
                            episode=[s,a,next_s,r]
                        elif self.state_name==None:
                            episode=[s,self.action_name[a],next_s,r]
                        elif self.action_name==None:
                            episode=[self.state_name[s],a,self.state_name[next_s],r]
                        else:
                            episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                    break
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
                loss=self._train()
                t2=time.time()
                self.time+=(t2-t1)
        else:
            for _ in range(self.episode_step):
                t1=time.time()
                self.a+=1
                if type(self.nn.nn)!=list:
                    try:
                        if self.nn.explore!=None:
                            pass
                        action_prob=self.epsilon_greedy_policy(s,self.action_one)
                        a=np.random.choice(self.action,p=action_prob)
                        if self.action_name==None:
                            next_s,r,end=self.nn.explore(a)
                        else:
                            next_s,r,end=self.nn.explore(self.action_name[a])
                    except AttributeError:
                        action_prob=self.epsilon_greedy_policy(s,self.action_one)
                        a=np.random.choice(self.action,p=action_prob)
                        next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
                else:
                    try:
                        if self.nn.explore!=None:
                            pass 
                        if len(self.nn.param)==4:
                            if self.state_name==None:
                                a=(self.nn.nn[1](s,p=1)+self.core.random.normal([1])).numpy()
                            else:
                                a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+self.core.random.normal([1])).numpy()
                        else:
                            if self.state_name==None:
                                a=self.nn.nn[1](s).numpy()
                            else:
                                a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                        if len(a.shape)>0:
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.nn.explore(self.action_name[a])
                        else:
                            next_s,r,end=self.nn.explore(a)
                    except AttributeError:
                        if len(self.nn.param)==4:
                            a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+self.core.random.normal([1])).numpy()
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                        if len(a.shape)>0:
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
                        else:
                            next_s,r,end=self.nn.transition(self.state_name[s],a)
                if self.state_pool==None:
                    self.state_pool=self.core.expand_dims(self.state[self.state_name[s]],axis=0)
                    self.action_pool=self.core.expand_dims(a,axis=0)
                    self.next_state_pool=self.core.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool=self.core.expand_dims(r,axis=0)
                else:
                    self.state_pool=self.core.concat([self.state_pool,self.core.expand_dims(self.state[self.state_name[s]],axis=0)],0)
                    self.action_pool=self.core.concat([self.action_pool,self.core.expand_dims(a,axis=0)],0)
                    self.next_state_pool=self.core.concat([self.next_state_pool,self.core.expand_dims(self.state[self.state_name[next_s]],axis=0)],0)
                    self.reward_pool=self.core.concat([self.reward_pool,self.core.expand_dims(r,axis=0)],0)
                if len(self.state_pool)>self.pool_size:
                    self.state_pool=self.state_pool[1:]
                    self.action_pool=self.action_pool[1:]
                    self.next_state_pool=self.next_state_pool[1:]
                    self.reward_pool=self.reward_pool[1:]
                if end:
                    if self.save_episode==True:
                        if self.state_name==None and self.action_name==None:
                            episode=[s,a,next_s,r]
                        elif self.state_name==None:
                            episode=[s,self.action_name[a],next_s,r]
                        elif self.action_name==None:
                            episode=[self.state_name[s],a,self.state_name[next_s],r]
                        else:
                            episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                    break
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
                loss=self._train()
                t2=time.time()
                self.time+=(t2-t1)
        return loss,episode,end
    
    
    def train(self,episode_num,save=None,one=True,p=None,s=None):
        self.train_flag=True
        self.train_counter+=1
        if self.p==None:
            self.p=9
        else:
            self.p=p-1
        if self.s==None:
            self.s=1
            self.file_list=None
        else:
            self.s=s-1
            self.file_list=[]
        loss=0
        if episode_num!=None:
            for i in range(episode_num):
                if self.stop==True:
                    self.stop_func()
                loss,episode,end=self.train_(episode_num,i)
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
                        print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                        print()
                    else:
                        print('episode num:{0}   loss:{1:.6f}'.format(self.total_episode,loss))
                        print()
                if save!=None and i%s==0:
                    self.save(self.total_episode,one)
                if self.save_episode==True:
                    if end:
                        episode.append('end')
                    self.episode.append(episode)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                if self.end_loss!=None and loss<=self.end_loss:
                    self.nn.param=self.param
                    self.param=None
                    break
        elif self.ol==None:
            i=0
            while True:
                if self.stop==True:
                    self.stop_func()
                loss,episode,end=self.train_(episode_num,i)
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
                        print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                        print()
                    else:
                        print('episode num:{0}   loss:{1:.6f}'.format(self.total_episode,loss))
                        print()
                if save!=None and i%s==0:
                    self.save(self.total_episode,one)
                if self.save_episode==True:
                    if end:
                        episode.append('end')
                    self.episode.append(episode)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                if self.end_loss!=None and loss<=self.end_loss:
                    self.nn.param=self.param
                    self.param=None
                    break
        else:
            data=self.ol()
            loss=self.opt_t(data)
            if self.thread_lock!=None:
                self.thread_lock.acquire()
                loss=loss.numpy()
                self.nn.train_loss.append(loss.astype(np.float32))
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                self.total_episode+=1
                self.thread_lock.release()
            else:
                loss=loss.numpy()
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
        print()
        print('time:{0}s'.format(self.time))
        self.train_flag=False
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
            self.train_flag=False
            self.save(self.total_episode,True)
            print('\nSystem have stopped training,Neural network have been saved.')
            return
        else:
            print('\nSystem have stopped training.')
            return
    
        
    def _save(self):
        if self.save_epi==self.total_episode:
            self.save(self.total_episode,False)
            self.save_epi=None
            print('\nNeural network have saved and training have suspended.')
            return
        elif self.save_epi!=None and self.save_epi>self.total_episode:
            print('\nsave_epoch>total_epoch')
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
    
    
    def save_p(self):
        parameter_file=open('param.dat','wb')
        pickle.dump(self.nn.param,parameter_file)
        parameter_file.close()
        return
    
    
    def save_e(self):
        episode_file=open('episode.dat','wb')
        pickle.dump(self.episode,episode_file)
        episode_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if one==True:
            output_file=open('save.dat','wb')
            parameter_file=open('param.dat','wb')
            if self.save_episode==True:
                episode_file=open('episode.dat','wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
        else:
            output_file=open('save-{0}.dat'.format(i),'wb')
            parameter_file=open('param-{0}.dat'.format(i),'wb')
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
        self.episode_num=self.epi_num
        pickle.dump(self.nn.param,parameter_file)
        if self.train_flag==False:
            self.nn.param=None
        self.nn.opt=None
        pickle.dump(self.nn,output_file)
        pickle.dump(self.opt.get_config(),output_file)
        pickle.dump(self.ol,output_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.action_len,output_file)
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
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.a,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.s,output_file)
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
        param=pickle.load(parameter_file)
        self.nn=pickle.load(input_file)
        self.nn.param=param
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.config=pickle.load(input_file)
        self.ol=pickle.load(input_file)
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
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
        self.loss_list=pickle.load(input_file)
        self.a=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.s=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
