from tensorflow import data as tf_data
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle


class kernel:
    def __init__(self,nn=None,thread=None,thread_lock=None,state=None,state_name=None,action_name=None,save_episode=True):
        self.nn=nn
        try:
            self.nn.km=1
        except AttributeError:
            pass
        if thread!=None:
            self.running_flag=np.array(0,dtype='int8')
            self.thread_num=np.arange(thread)
            self.thread_num=list(self.thread_num)
            self.reward=np.zeros(thread)
            self.loss=np.zeros(thread)
            self.sc=np.zeros(thread)
        self.state_pool=[]
        self.action_pool=[]
        self.next_state_pool=[]
        self.reward_pool=[]
        self.done_pool=[]
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.suspend=False
        self.stop=None
        self.save_flag=None
        self.stop_flag=1
        self.continuance_flag=False
        self.end_loss=None
        self.thread=thread
        self.thread_counter=0
        self.thread_lock=thread_lock
        self.probability_list=[]
        self.running_flag_list=[]
        self.finish_list=[]
        self.PN=True
        self.save_episode=save_episode
        self.reward_list=[]
        self.loss_list=[]
        self.continuance_flag=False
        self.total_episode=0
        self.total_time=0
    
    
    def action_init(self,action_num=None,dtype=np.int32):
        if self.action_name!=None:
            self.action_num=len(self.action_name)
        else:
            action_num=self.action_num
            self.action_num=action_num
        if self.action_name!=None and len(self.action_name)>self.action_num:
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_num,dtype=dtype)))
        elif self.action_name!=None:
            if self.epsilon!=None:
                self.action_one=np.ones(len(self.action_name),dtype=dtype)
        if self.action_num>action_num:
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(self.action_num-action_num,dtype=dtype)))
        else:
            if self.epsilon!=None:
                self.action_one=np.ones(self.action_num,dtype=dtype)
        return
    
    
    def add_threads(self,thread):
        thread_num=np.arange(thread)+self.thread
        self.thread_num=self.thread_num.extend(thread_num)
        self.thread+=thread
        self.sc=np.concatenate((self.sc,np.zeros(thread)))
        self.reward=np.concatenate((self.reward,np.zeros(thread)))
        self.loss=np.concatenate((self.loss,np.zeros(thread)))
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_num=None,criterion=None,end_loss=None,init=None):
        if type(epsilon)!=list and epsilon!=None:
            self.epsilon=np.ones(self.thread)*epsilon
        elif epsilon==None:
            self.epsilon=np.zeros(self.thread)
        else:
            self.epsilon=epsilon
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
        if init==True:
            self.thread_counter=0
            self.thread_num=np.arange(self.thread)
            self.thread_num=list(self.thread_num)
            self.probability_list=[]
            self.running_flag=np.array(0,dtype='int8')
            self.running_flag_list=[]
            self.finish_list=[]
            self.PN=True
            self.episode=[]
            self.epsilon=[]
            self.state_pool=[]
            self.action_pool=[]
            self.next_state_pool=[]
            self.reward_pool=[]
            self.reward=np.zeros(self.thread)
            self.loss=np.zeros(self.thread)
            self.reward_list=[]
            self.loss_list=[]
            self.sc=np.zeros(self.thread)
            self.total_episode=0
            self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one,epsilon):
        action_prob=action_one*epsilon/len(action_one)
        if self.state==None:
            best_a=np.argmax(self.nn.nn(s))
        else:
            best_a=np.argmax(self.nn.nn(self.state[self.state_name[s]]))
        action_prob[best_a.numpy()]+=1-epsilon
        return action_prob
    
    
    def get_episode(self,max_step=None):
        counter=0
        episode=[]
        s=self.nn.env.reset()
        self.end_flag=False
        while True:
            try:
                if self.nn.nn!=None:
                    pass
                try:
                    if self.nn.env!=None:
                        pass
                    try:
                        if self.nn.action!=None:
                            pass
                        s=np.expand_dims(s,axis=0)
                        a=self.nn.action(s)
                    except AttributeError:
                        s=np.expand_dims(s,axis=0)
                        a=np.argmax(self.nn.nn(s)).numpy()
                    if self.action_name==None:
                        next_s,r,done=self.nn.env(a)
                    else:
                        next_s,r,done=self.nn.env(self.action_name[a])
                except AttributeError:
                    a=np.argmax(self.nn.nn(s)).numpy()
                    next_s,r,done=self.nn.transition(self.state_name[s],self.action_name[a])
            except AttributeError:
                try:
                    if self.nn.env!=None:
                        pass
                    if self.state_name==None:
                        s=np.expand_dims(s,axis=0)
                        a=self.nn.actor(s).numpy()
                    else:
                        a=self.nn.actor(self.state[self.state_name[s]]).numpy()
                    a=np.squeeze(a)
                    next_s,r,done=self.nn.env(a)
                except AttributeError:
                    a=self.nn.actor(self.state[self.state_name[s]]).numpy()
                    a=np.squeeze(a)
                    next_s,r,done=self.nn.transition(self.state_name[s],a)
            try:
                if self.nn.stop!=None:
                    pass
                if self.nn.stop(next_s):
                    break
            except AttributeError:
                pass
            if self.end_flag==True:
                break
            elif done:
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
            else:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append([s,self.action_name[a],next_s,r])
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
            if max_step!=None and counter==max_step-1:
                break
            s=next_s
            counter+=1
        return episode
    
    
    def pool(self,s,a,next_s,r,done,t,index):
        if self.PN==True:
            self.thread_lock[0].acquire()
            if self.state_pool[index]==None and type(self.state_pool[index])!=np.ndarray:
                if self.state==None:
                    self.state_pool[index]=s
                    if type(a)==int:
                        a=np.array(a,np.int64)
                        self.action_pool[index]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[index]=a
                    self.next_state_pool[index]=np.expand_dims(next_s,axis=0)
                    self.reward_pool[index]=np.expand_dims(r,axis=0)
                    self.done_pool[index]=np.expand_dims(done,axis=0)
                else:
                    self.state_pool[index]=np.expand_dims(self.state[self.state_name[s]],axis=0)
                    if type(a)==int:
                        a=np.array(a,np.int64)
                        self.action_pool[index]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[index]=a
                    self.next_state_pool[index]=np.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool[index]=np.expand_dims(r,axis=0)
                    self.done_pool[index]=np.expand_dims(done,axis=0)
            else:
                try:
                    if self.state==None:
                        self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)
                        if type(a)==int:
                            a=np.array(a,np.int64)
                            self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                        else:
                            self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                        self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)
                        self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                        self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                    else:
                        self.state_pool[index]=np.concatenate((self.state_pool[index],np.expand_dims(self.state[self.state_name[s]],axis=0)),0)
                        if type(a)==int:
                            a=np.array(a,np.int64)
                            self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                        else:
                            self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                        self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(self.state[self.state_name[next_s]],axis=0)),0)
                        self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                        self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                except:
                    pass
            self.thread_lock[0].release()
        else:
            if self.state_pool[t]==None and type(self.state_pool[t])!=np.ndarray:
                if self.state==None:
                    self.state_pool[t]=s
                    if type(a)==int:
                        a=np.array(a,np.int64)
                        self.action_pool[t]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[t]=a
                    self.next_state_pool[t]=np.expand_dims(next_s,axis=0)
                    self.reward_pool[t]=np.expand_dims(r,axis=0)
                    self.done_pool[t]=np.expand_dims(done,axis=0)
                else:
                    self.state_pool[t]=np.expand_dims(self.state[self.state_name[s]],axis=0)
                    if type(a)==int:
                        a=np.array(a,np.int64)
                        self.action_pool[t]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[t]=a
                    self.next_state_pool[t]=np.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool[t]=np.expand_dims(r,axis=0)
                    self.done_pool[t]=np.expand_dims(done,axis=0)
            else:
                if self.state==None:
                    self.state_pool[t]=np.concatenate((self.state_pool[t],s),0)
                    if type(a)==int:
                        a=np.array(a,np.int64)
                        self.action_pool[t]=np.concatenate((self.action_pool[t],np.expand_dims(a,axis=0)),0)
                    else:
                        self.action_pool[t]=np.concatenate((self.action_pool[t],a),0)
                    self.next_state_pool[t]=np.concatenate((self.next_state_pool[t],np.expand_dims(next_s,axis=0)),0)
                    self.reward_pool[t]=np.concatenate((self.reward_pool[t],np.expand_dims(r,axis=0)),0)
                    self.done_pool[t]=np.concatenate((self.done_pool[t],np.expand_dims(done,axis=0)),0)
                else:
                    self.state_pool[t]=np.concatenate((self.state_pool[t],np.expand_dims(self.state[self.state_name[s]],axis=0)),0)
                    if type(a)==int:
                        a=np.array(a,np.int64)
                        self.action_pool[t]=np.concatenate((self.action_pool[t],np.expand_dims(a,axis=0)),0)
                    else:
                        self.action_pool[t]=np.concatenate((self.action_pool[t],a),0)
                    self.next_state_pool[t]=np.concatenate((self.next_state_pool[t],np.expand_dims(self.state[self.state_name[next_s]],axis=0)),0)
                    self.reward_pool[t]=np.concatenate((self.reward_pool[t],np.expand_dims(r,axis=0)),0)
                    self.done_pool[t]=np.concatenate((self.done_pool[t],np.expand_dims(done,axis=0)),0)
        if self.state_pool[t]!=None and len(self.state_pool[t])>self.pool_size:
            self.state_pool[t]=self.state_pool[t][1:]
            self.action_pool[t]=self.action_pool[t][1:]
            self.next_state_pool[t]=self.next_state_pool[t][1:]
            self.reward_pool[t]=self.reward_pool[t][1:]
            self.done_pool[t]=self.done_pool[t][1:]
        return
    
    
    def env(self,s,epsilon,t):
        try:
            if self.nn.nn!=None:
                pass
            try:
                if self.nn.env!=None:
                    pass
                s=np.expand_dims(s,axis=0)
                if self.epsilon==None:
                    self.epsilon[t]=self.nn.epsilon(self.sc[t],t)
                try:
                    if self.nn.action!=None:
                        pass
                    a=self.nn.action(s)
                except AttributeError:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action_num,p=action_prob)
                if self.action_name==None:
                    next_s,r,done=self.nn.env(a)
                else:
                    next_s,r,done=self.nn.env(self.action_name[a])
            except AttributeError:
                if self.epsilon==None:
                    self.epsilon[t]=self.nn.epsilon(self.sc[t],t)
                try:
                    if self.nn.action!=None:
                        pass
                    a=self.nn.action(s)
                except AttributeError:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action_num,p=action_prob)
                next_s,r,done=self.nn.transition(self.state_name[s],self.action_name[a])
        except AttributeError:
            try:
                if self.nn.env!=None:
                    pass 
                if self.state_name==None:
                    s=np.expand_dims(s,axis=0)
                    a=(self.nn.actor(s)+self.nn.noise()).numpy()
                else:
                    a=(self.nn.actor(self.state[self.state_name[s]])+self.nn.noise()).numpy()
                next_s,r,done=self.nn.env(a)
            except AttributeError:
                a=(self.nn.actor(self.state[self.state_name[s]])+self.nn.noise()).numpy()
                next_s,r,done=self.nn.transition(self.state_name[s],a)
        if self.PN==True:
            while len(self.running_flag_list)<t:
                pass
            if len(self.running_flag_list)==t:
                self.thread_lock[2].acquire()
                self.running_flag_list.append(self.running_flag[1:])
                self.thread_lock[2].release()
            else:
                if len(self.running_flag_list[t])<self.thread_counter or np.sum(self.running_flag_list[t])>self.thread_counter:
                    self.running_flag_list[t]=self.running_flag[1:]
            while len(self.probability_list)<t:
                pass
            if len(self.probability_list)==t:
                self.thread_lock[2].acquire()
                self.probability_list.append(np.array(self.running_flag_list[t],dtype=np.float16)/np.sum(self.running_flag_list[t]))
                self.thread_lock[2].release()
            else:
                if len(self.probability_list[t])<self.thread_counter or np.sum(self.running_flag_list[t])>self.thread_counter:
                    self.probability_list[t]=np.array(self.running_flag_list[t],dtype=np.float16)/np.sum(self.running_flag_list[t])
            while True:
                index=np.random.choice(len(self.probability_list[t]),p=self.probability_list[t])
                if index in self.finish_list:
                    continue
                else:
                    break
        else:
            index=None
        self.pool(s,a,next_s,r,done,t,index)
        if self.save_episode==True:
            if self.state_name==None and self.action_name==None:
                episode=[s,a,next_s,r]
            elif self.state_name==None:
                episode=[s,self.action_name[a],next_s,r]
            elif self.action_name==None:
                episode=[self.state_name[s],a,self.state_name[next_s],r]
            else:
                episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
        return next_s,r,done,episode,index
    
    
    def end(self):
        if self.end_loss!=None and self.loss_list[-1]<=self.end_loss:
            return True
    
    
    def opt_t(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        return loss.numpy()
    
    
    def _train(self,t,j=None,batches=None,length=None):
        if length%self.batch!=0:
            try:
                if self.nn.data_func!=None:
                    pass
                state_batch,action_batch,next_state_batch,reward_batch=self.nn.data_func(self.state_pool[t],self.action_pool[t],self.next_state_pool[t],self.reward_pool[t],self.done_pool[t],self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
            except AttributeError:
                index1=batches*self.batch
                index2=self.batch-(length-batches*self.batch)
                state_batch=np.concatenate((self.state_pool[t][index1:length],self.state_pool[t][:index2]),0)
                action_batch=np.concatenate((self.action_pool[t][index1:length],self.action_pool[t][:index2]),0)
                next_state_batch=np.concatenate((self.next_state_pool[t][index1:length],self.next_state_pool[t][:index2]),0)
                reward_batch=np.concatenate((self.reward_pool[t][index1:length],self.reward_pool[t][:index2]),0)
                done_batch=np.concatenate((self.done_pool[t][index1:length],self.done_pool[t][:index2]),0)
            loss=self.opt_t(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            self.loss[t]+=loss
            try:
                self.nn.bc[t]+=1
            except AttributeError:
                pass
            return
        try:
            if self.nn.data_func!=None:
                pass
            state_batch,action_batch,next_state_batch,reward_batch=self.nn.data_func(self.state_pool[t],self.action_pool[t],self.next_state_pool[t],self.reward_pool[t],self.done_pool[t],self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
        except AttributeError:
            index1=j*self.batch
            index2=(j+1)*self.batch
            state_batch=self.state_batch[t][index1:index2]
            action_batch=self.action_batch[t][index1:index2]
            next_state_batch=self.next_state_batch[t][index1:index2]
            reward_batch=self.reward_batch[t][index1:index2]
            done_batch=self.done_batch[t][index1:index2]
            loss=self.opt_t(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            self.loss[t]+=loss
        try:
            self.nn.bc[t]=j
        except AttributeError:
            pass
        return
    
    
    def train_(self,t):
        train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool[t],self.action_pool[t],self.next_state_pool[t],self.reward_pool[t],self.done_pool[t])).shuffle(len(self.state_pool[t])).batch(self.batch)
        for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
            if self.stop==True:
                if self.stop_func() or self.stop_flag==0:
                    return
            self.suspend_func()
            loss=self.opt_t(state_batch,action_batch,next_state_batch,reward_batch)
            self.loss[t]+=loss
            try:
                self.nn.bc[t]+=1
            except AttributeError:
                pass
        return
            
    
    def _train_(self,t):
        if len(self.done_pool[t])<self.batch:
            return
        else:
            self.loss[t]=0
            if self.PN==True:
                length=min(len(self.state_pool[t]),len(self.action_pool[t]),len(self.next_state_pool[t]),len(self.reward_pool[t]),len(self.done_pool[t]))
                batches=int((length-length%self.batch)/self.batch)
                if length%self.batch!=0:
                    batches+=1
                for j in range(batches):
                    if self.stop==True:
                        if self.stop_func() or self.stop_flag==0:
                            return
                    self.suspend_func()
                    self._train(t,j,batches,length)
            else:
                try:
                    self.nn.bc[t]=0
                except AttributeError:
                    pass
                self.train_(t)
            if self.PN==True:
                self.thread_lock[2].acquire()
            else:
                self.thread_lock[1].acquire()
            if self.update_step!=None:
                if self.sc[t]+1%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
            if self.PN==True:
                self.thread_lock[2].release()
            else:
                self.thread_lock[1].release()
            self.loss[t]=self.loss[t]/batches
            self.loss[t]=self.loss[t].astype(np.float32)
        self.sc[t]+=1
        try:
            self.nn.ec[t]+=1
        except AttributeError:
            pass
        return
    
    
    def train(self,episode_num):
        try:
            t=self.thread_num.pop(0)
        except IndexError:
            print('\nError,please add thread.')
            return
        if t in self.finish_list:
            return
        while self.state_pool!=None and len(self.state_pool)<t:
            pass
        if self.state_pool!=None and len(self.state_pool)==t:
            if self.PN==True:
                self.thread_lock[3].acquire()
            else:
                self.thread_lock[0].acquire()
            self.state_pool.append(None)
            self.action_pool.append(None)
            self.next_state_pool.append(None)
            self.reward_pool.append(None)
            self.running_flag=np.append(self.running_flag,np.array(1,dtype='int8'))
            self.thread_counter+=1
            self.finish_list.append(None)
            try:
                self.nn.ec.append(0)
            except AttributeError:
                pass
            try:
                self.nn.bc.append(0)
            except AttributeError:
                pass
            if self.PN==True:
                self.thread_lock[3].release()
            else:
                self.thread_lock[0].release()
        for k in range(episode_num):
            if self.stop==True:
                if self.stop_func() or self.stop_flag==0:
                    return
            episode=[]
            if self.state_name==None:
                s=self.nn.env(initial=True)
            else:
                s=self.nn.transition(initial=True)
            if self.episode_step==None:
                while True:
                    if self.stop==True:
                        if self.stop_func() or self.stop_flag==0:
                            self.finish_list[t]=t
                            return
                    try:
                        epsilon=self.epsilon[t]
                    except:
                        pass
                    next_s,r,done,_episode,index=self.env(s,epsilon,t)
                    self.reward[t]+=r
                    s=next_s
                    if self.state_pool[t]!=None and self.action_pool[t]!=None and self.next_state_pool[t]!=None and self.reward_pool[t]!=None and self.done_pool[t]!=None:
                        self._train_(t)
                    if self.stop_flag==0:
                        self.finish_list[t]=t
                        return
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        if self.PN==True:
                            self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.total_episode+=1
                        self.loss_list.append(self.loss[t])
                        if self.PN==True:
                            self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
            else:
                for l in range(self.episode_step):
                    if self.stop==True:
                        if self.stop_func() or self.stop_flag==0:
                            self.finish_list[t]=t
                            return
                    try:
                        epsilon=self.epsilon[t]
                    except:
                        pass
                    next_s,r,done,episode,index=self.env(s,epsilon,t)
                    self.reward[t]+=r
                    s=next_s
                    if self.state_pool[t]!=None and self.action_pool[t]!=None and self.next_state_pool[t]!=None and self.reward_pool[t]!=None and self.done_pool[t]!=None:
                        self._train_(t)
                    if self.stop_flag==0:
                        self.finish_list[t]=t
                        return
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        if self.PN==True:
                            self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.total_episode+=1
                        self.loss_list.append(self.loss[t])
                        if self.PN==True:
                            self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
                    if l==self.episode_step-1:
                        if self.PN==True:
                            self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.total_episode+=1
                        self.loss_list.append(self.loss[t])
                        if self.PN==True:
                            self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
            if self.PN==True:
                self.thread_lock[3].acquire()
            else:
                self.thread_lock[0].acquire()
            self.reward_list.append(self.reward[t])
            self.reward[t]=0
            if self.save_episode==True:
                self.episode.append(episode)
            if self.PN==True:
                self.thread_lock[3].release()
            else:
                self.thread_lock[0].release()
        if self.PN==True:
            self.running_flag[t+1]=0
            self.thread_lock[3].acquire()
            if t not in self.finish_list:
                self.finish_list[t]=t
            self.thread_counter-=1
            self.thread_lock[3].release()
            self.state_pool[t]=None
            self.action_pool[t]=None
            self.next_state_pool[t]=None
            self.reward_pool[t]=None
        return
    
    
    def suspend_func(self):
        if self.suspend==True:
            while True:
                if self.suspend==False:
                    break
        return
    
    
    def stop_func(self):
        if self.trial_num!=None:
            if len(self.reward_list)>=self.trial_num:
                avg_reward=statistics.mean(self.reward_list[-self.trial_num:])
                if self.criterion!=None and avg_reward>=self.criterion:
                    if self.PN==True:
                        self.thread_lock[4].acquire()
                    else:
                        self.thread_lock[2].acquire()
                    self.save(self.total_episode,True)
                    self.save_flag=True
                    if self.PN==True:
                        self.thread_lock[4].release()
                    else:
                        self.thread_lock[2].release()
                    self.stop_flag=0
                    return True
        elif self.end():
            if self.PN==True:
                self.thread_lock[4].acquire()
            else:
                self.thread_lock[2].acquire()
            self.save(self.total_episode,True)
            self.save_flag=True
            if self.PN==True:
                self.thread_lock[4].release()
            else:
                self.thread_lock[2].release()
            self.stop_flag=0
            return True
        elif self.stop_flag==1:
            self.stop_flag=0
            return True
        return False
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_list)),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.loss_list)),self.loss_list)
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
    
    
    def save(self):
        if self.save_flag==True:
            return
        output_file=open('save.dat','wb')
        if self.save_episode==True:
            episode_file=open('episode.dat','wb')
            pickle.dump(self.episode,episode_file)
            episode_file.close()
        pickle.dump(self.nn,output_file)
        pickle.dump(self.thread,output_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.done_pool,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.sc,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.thread_counter,output_file)
        pickle.dump(self.probability_list,output_file)
        pickle.dump(self.running_flag,output_file)
        pickle.dump(self.running_flag_list,output_file)
        pickle.dump(self.finish_list,output_file)
        pickle.dump(self.PN,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        if self.save_flag==True:
            print('\nSystem have stopped,Neural network have saved.')
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
        self.thread=pickle.load(input_file)
        if self.continuance_flag==True:
            self.thread_num=np.arange(self.thread)
            self.thread_num=list(self.thread_num)
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.done_pool=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.sc=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.thread_counter=pickle.load(input_file)
        self.probability_list=pickle.load(input_file)
        self.running_flag=pickle.load(input_file)
        self.running_flag_list=pickle.load(input_file)
        self.finish_list=pickle.load(input_file)
        self.PN=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
