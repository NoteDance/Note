import tensorflow as tf
from tensorflow import data as tf_data
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sys import getsizeof
import pickle
import os


class kernel:
    def __init__(self,nn=None,thread=None,save_episode=False):
        self.nn=nn
        try:
            self.nn.km=1
        except AttributeError:
            pass
        if thread!=None:
            self.thread_num=np.arange(thread)
            self.thread_num=list(self.thread_num)
            self.reward=np.zeros(thread,dtype=np.float32)
            self.loss=np.zeros(thread,dtype=np.float32)
            self.sc=np.zeros(thread,dtype=np.float32)
            self.opt_counter=np.zeros(thread,dtype=np.float32)
        self.threading=None
        self.gradient_lock=None
        self.max_lock=None
        self.row=None
        self.rank=None
        self.d_index=0
        self.state_pool={}
        self.action_pool={}
        self.next_state_pool={}
        self.reward_pool={}
        self.done_pool={}
        self.episode=[]
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.episode_num=0
        self.update_step=None
        self.trial_num=None
        self.running_list=[]
        self.suspend=False
        self.suspend_list=[]
        self.suspended_list=[]
        self.stop=False
        self.stop_list=[]
        self.stopped_list=[]
        self.stop_list_m=[]
        self.save_flag=False
        self.stop_flag=1
        self.add_flag=False
        self.memory_flag=False
        self.memory_priority=False
        self.episode_list=np.array(0,dtype=np.int8)
        self.episode_list_copy=None
        self.param_memory=0
        self.grad_memory=0
        self.c_memory=0
        self.max_memory=0
        self.grad_memory_list=[]
        self.pool_memory_list=[]
        self.episode_memory_list=[]
        self.episode_memory_t_value=None
        self.memory_t_value=None
        self.end_loss=None
        self.thread=thread
        self.thread_counter=0
        self.thread_lock=None
        self.pool_lock=[]
        self.probability_list=[]
        self.running_flag_list=[]
        self.index_matrix=[]
        self.one_matrix=[]
        self.row_list=[]
        self.row_sum_list=[]
        self.rank_sum_list=[]
        self.row_probability=[]
        self.rank_probability=[]
        self.direction_index=0
        self.finish_list=[]
        try:
            if self.nn.row!=None:
                self.row_one=np.array(0,dtype=np.int8)
                self.rank_one=np.array(0,dtype=np.int8)
        except AttributeError:
            self.running_flag=np.array(0,dtype=np.int8)
        self.PN=True
        self.PO=None
        self.max_episode_num=None
        self.save_episode=save_episode
        self.ln_list=[]
        self.gradient_list=[]
        self.exception_list=[]
        self.muti_p=None
        self.muti_s=None
        self.muti_save=1
        self.reward_list=[]
        self.loss_list=[]
        self.total_episode=0
        self.total_time=0
    
    
    def action_vec(self):
        self.action_one=np.ones(self.action_num,dtype=np.int8)
        return
    
    
    def calculate_memory(self,s=None,a=None,next_s=None,r=None,done=None,t=None):
        if s!=None:
            self.episode_memory_list[t]+=getsizeof(s)
            self.episode_memory_list[t]+=getsizeof(a)
            self.episode_memory_list[t]+=getsizeof(next_s)
            self.episode_memory_list[t]+=getsizeof(r)
            return
        if self.memory_flag==True:
            for i in range(self.nn.param):
                self.param_memory+=getsizeof(self.nn.param[i])
            self.grad_memory=self.param_memory
            if self.PO==1 or self.PO==2:
                self.max_memory=self.data_memory+self.param_memory+self.grad_memory
            elif self.PO==3:
                if self.row!=None:
                    self.max_memory=self.data_memory+self.param_memory+self.grad_memory*self.row*self.rank
                elif self.max_lock!=None:
                    self.max_memory=self.data_memory+self.param_memory+self.grad_memory*self.max_lock
                else:
                    self.max_memory=self.data_memory+self.param_memory+self.grad_memory*len(self.gradient_lock)
            return
    
    
    def calculate_memory_(self,t,ln=None):
        if self.PO==3:
            self.grad_memory_list[ln]=self.grad_memory
        self.pool_memory_list[t]=getsizeof(self.state_pool[t][0])*len(self.state_pool[t])+getsizeof(self.action_pool[t][0])*len(self.action_pool[t])+getsizeof(self.next_state_pool[t][0])*len(self.next_state_pool[t])+getsizeof(self.reward_pool[t][0])*len(self.reward_pool[t])+getsizeof(self.done_pool[t][0])*len(self.done_pool[t])
        if self.save_episode==False:
            if self.PO==3:
                self.c_memory=self.data_memory+self.param_memory+sum(self.grad_memory_list)+sum(self.pool_memory_list)
            else:
                self.c_memory=self.data_memory+self.param_memory+self.grad_memory+sum(self.pool_memory_list)
        else:
            episode_memory=sum(self.episode_memory_list)
            if self.PO==3:
                self.c_memory=self.data_memory+self.param_memory+sum(self.grad_memory_list)+sum(self.pool_memory_list)+episode_memory
            else:
                self.c_memory=self.data_memory+self.param_memory+self.grad_memory+sum(self.pool_memory_list)+episode_memory
            if self.episode_memory_t_value!=None and episode_memory>self.episode_memory_t_value:
                self.save_episode=False
        return
    
    
    def calculate_memory_ol(self,ln=None):
        if self.memory_flag==True:
            self.grad_memory_list[ln]=self.grad_memory
            self.c_memory=self.data_memory+self.param_memory+sum(self.grad_memory_list)
        return
    
    
    def add_threads(self,thread,row=None,rank=None):
        if row!=None:
            self.thread=row*rank-self.nn.row*self.nn.rank
            self.nn.row=row
            self.nn.rank=rank
            self.add_flag=True
        thread_num=np.arange(thread)+self.thread
        self.thread_num=self.thread_num.extend(thread_num)
        self.thread+=thread
        self.sc=np.concatenate((self.sc,np.zeros(thread,dtype=np.float32)))
        self.reward=np.concatenate((self.reward,np.zeros(thread,dtype=np.float32)))
        self.loss=np.concatenate((self.loss,np.zeros(thread,dtype=np.float32)))
        self.opt_counter=np.concatenate((self.opt_counter,np.zeros(thread,dtype=np.float32)))
        return
    
    
    def init(self):
        self.suspend=False
        self.suspend_list=[]
        self.suspended_list=[]
        self.stop=None
        self.stop_list=[]
        self.stopped_list=[]
        self.save_flag=False
        self.stop_flag=1
        self.add_flag=False
        self.memory_flag=False
        self.param_memory=0
        self.grad_memory=0
        self.c_memory=0
        self.max_memory=0
        self.grad_memory_list=[]
        self.thread_counter=0
        self.thread_num=np.arange(self.thread)
        self.thread_num=list(self.thread_num)
        self.probability_list=[]
        self.running_flag=np.array(0,dtype=np.int8)
        self.running_flag_list=[]
        self.index_matrix=[]
        self.one_matrix=[]
        self.row_list=[]
        self.row_sum_list=[]
        self.rank_sum_list=[]
        self.row_probability=[]
        self.direction_index=0
        self.finish_list=[]
        try:
            if self.nn.row!=None:
                self.row_one=np.array(0,dtype=np.int8)
                self.rank_one=np.array(0,dtype=np.int8)
        except AttributeError:
            self.running_flag=np.array(0,dtype=np.int8)
        try:
            if self.nn.pr!=None:
                self.nn.pr.TD=[]
                self.nn.pr.index=[]
        except AttributeError:
            pass
        self.PN=True
        self.episode=[]
        self.epsilon=None
        self.state_pool={}
        self.action_pool={}
        self.next_state_pool={}
        self.reward_pool={}
        self.done_pool={}
        self.reward=np.zeros(self.thread,dtype=np.float32)
        self.loss=np.zeros(self.thread,dtype=np.float32)
        self.reward_list=[]
        self.loss_list=[]
        self.sc=np.zeros(self.thread,dtype=np.float32)
        self.total_episode=0
        self.total_time=0
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_num=None,criterion=None,end_loss=None):
        if epsilon!=None:
            self.epsilon=np.ones(self.thread)*epsilon
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
        if epsilon!=None:
            self.action_vec()
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon):
        action_prob=self.action_one*epsilon/len(self.action_one)
        best_a=np.argmax(self.nn.nn.fp(s))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def get_episode(self,max_step=None,seed=None):
        counter=0
        episode=[]
        if seed==None:
            s=self.nn.env.reset()
        else:
            s=self.nn.env.reset(seed=seed)
        self.end_flag=False
        while True:
            try:
                if self.nn.nn!=None:
                    try:
                        if self.nn.action!=None:
                            s=np.expand_dims(s,axis=0)
                            a=self.nn.action(s)
                    except AttributeError:
                        s=np.expand_dims(s,axis=0)
                        a=np.argmax(self.nn.nn.fp(s)).numpy()
                    next_s,r,done=self.nn.env(a)
            except AttributeError:
                s=np.expand_dims(s,axis=0)
                a=self.nn.actor.fp(s).numpy()
                a=np.squeeze(a)
                next_s,r,done=self.nn.env(a)
            try:
                if self.nn.stop!=None:
                    pass
                if self.nn.stop(next_s):
                    break
            except AttributeError:
                pass
            if self.end_flag==True:
                break
            if done:
                episode.append([s,a,next_s,r])
                episode.append('done')
                break
            else:
                episode.append([s,a,next_s,r])
            if max_step!=None and counter==max_step-1:
                break
            s=next_s
            counter+=1
        return episode
    
    
    def pool(self,s,a,next_s,r,done,t,index):
        if self.PN==True:
            self.pool_lock[index].acquire()
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None:
                self.state_pool[index]=s
                if type(a)==int:
                    a=np.array(a,np.int32)
                    self.action_pool[index]=np.expand_dims(a,axis=0)
                else:
                    self.action_pool[index]=a
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0)
                self.reward_pool[index]=np.expand_dims(r,axis=0)
                self.done_pool[index]=np.expand_dims(done,axis=0)
            else:
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)
                    if type(a)==int:
                        a=np.array(a,np.int64)
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                    else:
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                except:
                    pass
            try:
                if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size:
                    self.state_pool[index]=self.state_pool[index][1:]
                    self.action_pool[index]=self.action_pool[index][1:]
                    self.next_state_pool[index]=self.next_state_pool[index][1:]
                    self.reward_pool[index]=self.reward_pool[index][1:]
                    self.done_pool[index]=self.done_pool[index][1:]
                    del self.state_pool[t]
                    del self.action_pool[t]
                    del self.next_state_pool[t]
                    del self.reward_pool[t]
                    del self.done_pool[t]
            except:
                pass
            self.pool_lock[index].release()
        else:
            if type(self.state_pool[t])==np.ndarray and self.state_pool[t]==None:
                self.state_pool[t]=s
                if type(a)==int:
                    a=np.array(a,np.int32)
                    self.action_pool[t]=np.expand_dims(a,axis=0)
                else:
                    self.action_pool[t]=a
                self.next_state_pool[t]=np.expand_dims(next_s,axis=0)
                self.reward_pool[t]=np.expand_dims(r,axis=0)
                self.done_pool[t]=np.expand_dims(done,axis=0)
            else:
                self.state_pool[t]=np.concatenate((self.state_pool[t],s),0)
                if type(a)==int:
                    a=np.array(a,np.int64)
                    self.action_pool[t]=np.concatenate((self.action_pool[t],np.expand_dims(a,axis=0)),0)
                else:
                    self.action_pool[t]=np.concatenate((self.action_pool[t],a),0)
                self.next_state_pool[t]=np.concatenate((self.next_state_pool[t],np.expand_dims(next_s,axis=0)),0)
                self.reward_pool[t]=np.concatenate((self.reward_pool[t],np.expand_dims(r,axis=0)),0)
                self.done_pool[t]=np.concatenate((self.done_pool[t],np.expand_dims(done,axis=0)),0)
            if self.state_pool[t]!=None and len(self.state_pool[t])>self.pool_size:
                self.state_pool[t]=self.state_pool[t][1:]
                self.action_pool[t]=self.action_pool[t][1:]
                self.next_state_pool[t]=self.next_state_pool[t][1:]
                self.reward_pool[t]=self.reward_pool[t][1:]
                self.done_pool[t]=self.done_pool[t][1:]
        return
    
    
    def index_m(self,t):
        if self.add_flag==None and len(self.index_matrix)!=self.nn.row:
            if len(self.row_list)!=self.nn.rank:
                self.row_list.append(t)
                self.rank_one=np.append(self.rank_one,np.array(1,dtype=np.int8))
                if len(self.row_list)==self.nn.rank:
                    self.index_matrix.append(self.row_list.copy())
                    self.row_list=[]
                    self.one_matrix.append(self.rank_one)
                    self.rank_one=np.array(0,dtype=np.int8)
                    self.row_one=np.append(self.row_one,np.array(1,dtype=np.int8))
        elif self.add_flag==True:
            if len(self.index_matrix)!=self.nn.row:
                if self.direction_index>len(self.index_matrix) and self.row_list==[]:
                    self.index_matrix.append([])
                    self.one_matrix.append(np.array(0,dtype=np.int8))
                    self.row_one=np.append(self.row_one,np.array(1,dtype=np.int8))
                self.row_list.append(t)
                self.one_matrix[self.direction_index]=np.append(self.one_matrix[self.direction_index],np.array(1,dtype='int8'))
                if len(self.index_matrix[self.direction_index])+len(self.row_list)==self.nn.rank:
                    self.index_matrix[self.direction_index].extend(self.row_list.copy())
                    self.row_list=[]
                    self.direction_index+=1
                    if len(self.index_matrix)==self.nn.row and len(self.index_matrix[-1])==self.nn.rank:
                        self.direction_index=0
            else:
                self.row_list.append(t)
                self.one_matrix[self.direction_index]=np.append(self.one_matrix[self.direction_index],np.array(1,dtype='int8'))
                if len(self.index_matrix[self.direction_index])+len(self.row_list)==self.nn.rank:
                    self.index_matrix[self.direction_index].extend(self.row_list.copy())
                    self.row_list=[]
                    self.direction_index+=1
                    if len(self.index_matrix[-1])==self.nn.rank:
                        self.direction_index=0
        return
    
    
    def index(self,t):
        if self.PN==True:
            try:
                if self.nn.row!=None:
                    while True:
                        row_sum=np.sum(self.row_one)
                        if self.row_sum_list[t]==None:
                            self.row_sum_list[t]=row_sum
                        if self.row_sum_list[t]==row_sum:
                            row_index=np.random.choice(self.nn.row,p=self.row_probability[t])-1
                        else:
                            self.row_sum_list[t]=row_sum
                            self.row_probability[t]=self.row_one/row_sum
                            row_index=np.random.choice(self.nn.row,p=self.row_probability[t])-1
                        rank_sum=np.sum(self.one_matrix[row_index])
                        if rank_sum==0:
                            self.row_one[row_index]=0
                            continue
                        if self.rank_sum_list[t]==None:
                           self.rank_sum_list[t]=rank_sum
                        if self.rank_sum_list[t]==rank_sum:
                            rank_index=np.random.choice(self.nn.rank,p=self.rank_probability[t])-1
                        else:
                            self.rank_sum_list[t]=rank_sum
                            self.rank_probability[t]=self.one_matrix[row_index]/rank_sum
                            rank_index=np.random.choice(self.nn.rank,p=self.rank_probability[t])-1
                        index=self.index_matrix[row_index][rank_index]
                        if index in self.finish_list:
                            self.one_matrix[row_index][rank_index]=0
                            continue
                        else:
                            break
            except AttributeError:
                while len(self.running_flag_list)<t:
                    pass
                if len(self.running_flag_list)==t:
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[2].acquire()
                    else:
                        self.thread_lock[3].acquire()
                    self.running_flag_list.append(self.running_flag[1:].copy())
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[2].release()
                    else:
                        self.thread_lock[3].release()
                if len(self.running_flag_list[t])<self.thread_counter or np.sum(self.running_flag_list[t])>self.thread_counter:
                    self.running_flag_list[t]=self.running_flag[1:].copy()
                while len(self.probability_list)<t:
                    pass
                if len(self.probability_list)==t:
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[2].acquire()
                    else:
                        self.thread_lock[3].acquire()
                    self.probability_list.append(np.array(self.running_flag_list[t],dtype=np.float16)/np.sum(self.running_flag_list[t]))
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[2].release()
                    else:
                        self.thread_lock[3].release()
                self.probability_list[t]=np.array(self.running_flag_list[t],dtype=np.float16)/np.sum(self.running_flag_list[t])
                while True:
                    index=np.random.choice(len(self.probability_list[t]),p=self.probability_list[t])
                    if index in self.finish_list:
                        continue
                    else:
                        break
        else:
            index=None
        return index
    
    
    def env(self,s,epsilon,t):
        try:
            if self.nn.nn!=None:
                s=np.expand_dims(s,axis=0)
                if epsilon==None:
                    epsilon=self.nn.epsilon(self.sc[t],t)
                try:
                    if self.nn.action!=None:
                        try:
                            if self.nn.discriminator!=None:
                                a=self.nn.action(s)
                                reward=self.nn.discriminator(s,a)
                                s=np.squeeze(s)
                        except AttributeError:
                            a=self.nn.action(s).numpy()
                except AttributeError:
                    action_prob=self.epsilon_greedy_policy(s,epsilon)
                    a=np.random.choice(self.action_num,p=action_prob)
                next_s,r,done=self.nn.env(a)
        except AttributeError:
            s=np.expand_dims(s,axis=0)
            a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
            next_s,r,done=self.nn.env(a)
        index=self.index(t)
        r=np.array(r,dtype=np.float32)
        done=np.array(done,dtype=np.float32)
        try:
            if self.nn.pool!=None:
                self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,reward,done],t,index,self.pool_lock)
        except AttributeError:
            self.pool(s,a,next_s,r,done,t,index)
        if self.save_episode==True:
            episode=[s,a,next_s,r]
            self.calculate_memory(s,a,next_s,r,done,t)
            return next_s,r,done,episode,index
        else:
            return next_s,r,done,None,index
    
    
    def end(self):
        if self.end_loss!=None and self.loss_list[-1]<=self.end_loss:
            return True
    
    
    @tf.function
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,t):
        with tf.GradientTape(persistent=True) as tape:
            try:
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            except TypeError:
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
        try:
            if self.nn.attenuate!=None:
                self.opt_counter[t]=0
        except AttributeError:
            pass
        if self.PO==1:
            self.thread_lock[0].acquire()
            if self.episode_memory_t_value!=None and sum(self.episode_memory_list)>self.episode_memory_t_value:
                self.save_episode=False
            if self.memory_flag==True:
                self.calculate_memory_(t)
                if self.stop_func_m(self.thread_lock[0]):
                    return 0
                if self.stop_func_t_p(self.thread_lock[0],t):
                    return 0
            if self.stop_func_(self.thread_lock[0]):
                return 0
            try:
                gradient=self.nn.gradient(tape,loss)
                try:
                    if self.nn.attenuate!=None:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                except AttributeError:
                    pass
                try:
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                    try:
                        self.nn.opt(gradient)
                    except TypeError:
                        self.nn.opt(gradient,t)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                        try:
                            if self.nn.attenuate!=None:
                                gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                        except AttributeError:
                            pass
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                        actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                        critic_gradient=tape.gradient(loss[1],self.nn.param[1])
                        try:
                            if self.nn.attenuate!=None:
                                actor_gradient=self.nn.attenuate(actor_gradient,self.opt_counter,t)
                                critic_gradient=self.nn.attenuate(critic_gradient,self.opt_counter,t)
                        except AttributeError:
                            pass
                        self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                        self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.thread_lock[0].release()
        elif self.PO==2:
            self.thread_lock[0].acquire()
            if self.episode_memory_t_value!=None and sum(self.episode_memory_list)>self.episode_memory_t_value:
                self.save_episode=False
            if self.memory_flag==True:
                self.calculate_memory_(t)
                if self.stop_func_m(self.thread_lock[0]):
                    return 0
                if self.stop_func_t_p(self.thread_lock[0],t):
                    return 0
            if self.stop_func_(self.thread_lock[0]):
                return 0
            try:
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                except AttributeError:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            self.thread_lock[0].release()
            self.thread_lock[1].acquire()
            try:
                if self.nn.attenuate!=None:
                    try:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                    except NameError:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.opt_counter,t)
                        critic_gradient=self.nn.attenuate(critic_gradient,self.opt_counter,t)
            except AttributeError:
                pass
            try:
                if self.nn.gradient!=None:
                    try:
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                    except AttributeError:
                        try:
                            self.nn.opt(gradient)
                        except TypeError:
                            self.nn.opt(gradient,t)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                    self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                    self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.thread_lock[1].release()
        return loss
    
    
    @tf.function
    def opt_ol(self,data,t=None):
        with tf.GradientTape(persistent=True) as tape:
            try:
                loss=self.nn.loss(data)
            except:
                loss=self.nn.loss(data,t)
        if self.thread!=None:
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter[t]=0
            except AttributeError:
                pass
        if self.PO==1:
            self.thread_lock[0].acquire()
            try:
                gradient=self.nn.gradient(tape,loss)
                try:
                    if self.nn.attenuate!=None:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                except AttributeError:
                    pass
                try:
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except:
                    self.nn.opt(gradient)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                        try:
                            if self.nn.attenuate!=None:
                                gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                        except AttributeError:
                            pass
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                        actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                        critic_gradient=tape.gradient(loss[1],self.nn.param[1])
                        try:
                            if self.nn.attenuate!=None:
                                actor_gradient=self.nn.attenuate(actor_gradient,self.opt_counter,t)
                                critic_gradient=self.nn.attenuate(critic_gradient,self.opt_counter,t)
                        except AttributeError:
                            pass
                        self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                        self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.thread_lock[0].release()
        elif self.PO==2:
            self.thread_lock[0].acquire()
            try:
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                except AttributeError:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            self.thread_lock[0].release()
            self.thread_lock[1].acquire()
            try:
                if self.nn.attenuate!=None:
                    try:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                    except NameError:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.opt_counter,t)
                        critic_gradient=self.nn.attenuate(critic_gradient,self.opt_counter,t)
            except AttributeError:
                pass
            try:
                if self.nn.gradient!=None:
                    try:
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                    except AttributeError:
                        try:
                            self.nn.opt(gradient)
                        except TypeError:
                            self.nn.opt(gradient,t)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                    self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                    self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.thread_lock[1].release()
        return loss
    
    
    def _train(self,t,j=None,batches=None,length=None):
        if length%self.batch!=0:
            try:
                if self.nn.data_func!=None:
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool[t],self.action_pool[t],self.next_state_pool[t],self.reward_pool[t],self.done_pool[t],self.batch,t)
                    loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
            except AttributeError:
                index1=batches*self.batch
                index2=self.batch-(length-batches*self.batch)
                state_batch=np.concatenate((self.state_pool[t][index1:length],self.state_pool[t][:index2]),0)
                action_batch=np.concatenate((self.action_pool[t][index1:length],self.action_pool[t][:index2]),0)
                next_state_batch=np.concatenate((self.next_state_pool[t][index1:length],self.next_state_pool[t][:index2]),0)
                reward_batch=np.concatenate((self.reward_pool[t][index1:length],self.reward_pool[t][:index2]),0)
                done_batch=np.concatenate((self.done_pool[t][index1:length],self.done_pool[t][:index2]),0)
                loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
            self.loss[t]+=loss
            try:
                self.nn.bc[t]+=1
            except AttributeError:
                pass
            return
        try:
            if self.nn.data_func!=None:
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool[t],self.action_pool[t],self.next_state_pool[t],self.reward_pool[t],self.done_pool[t],self.batch,t)
                loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
        except AttributeError:
            index1=j*self.batch
            index2=(j+1)*self.batch
            state_batch=self.state_pool[t][index1:index2]
            action_batch=self.action_pool[t][index1:index2]
            next_state_batch=self.next_state_pool[t][index1:index2]
            reward_batch=self.reward_pool[t][index1:index2]
            done_batch=self.done_pool[t][index1:index2]
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
            self.loss[t]+=loss
        try:
            self.nn.bc[t]=j
        except AttributeError:
            pass
        return
    
    
    def train_(self,t):
        train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool[t],self.action_pool[t],self.next_state_pool[t],self.reward_pool[t],self.done_pool[t])).shuffle(len(self.state_pool[t])).batch(self.batch)
        for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
            if t in self.stop_list:
                return
            self.suspend_func(t)
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
            if self.stop_flag==0:
                return
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
                length=len(self.done_pool[t])
                batches=int((length-length%self.batch)/self.batch)
                if length%self.batch!=0:
                    batches+=1
                for j in range(batches):
                    if t in self.stop_list:
                        return
                    self.suspend_func(t)
                    self._train(t,j,batches,length)
                    if self.stop_flag==0:
                        return
            else:
                try:
                    self.nn.bc[t]=0
                except AttributeError:
                    pass
                self.train_(t)
                if self.stop_flag==0:
                    return
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.thread_lock[1].acquire()
                else:
                    self.thread_lock[2].acquire()
            else:
                self.thread_lock[1].acquire()
            if self.update_step!=None:
                if self.sc[t]%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.thread_lock[1].release()
                else:
                    self.thread_lock[2].release()
            else:
                self.thread_lock[1].release()
            self.loss[t]=self.loss[t]/batches
        self.sc[t]+=1
        try:
            self.nn.ec[t]+=1
        except AttributeError:
            pass
        return
    
    
    def train(self,episode_num):
        try:
            t=self.thread_num.pop(0)
            t=int(t)
        except IndexError:
            print('\nError,please add thread.')
            return
        if self.PN==True:
            if self.PO==1 or self.PO==3:
                self.thread_lock[2].acquire()
            else:
                self.thread_lock[3].acquire()
        else:
            self.thread_lock[0].acquire()
        self.state_pool[t]=None
        self.action_pool[t]=None
        self.next_state_pool[t]=None
        self.reward_pool[t]=None
        self.done_pool[t]=None
        try:
            if self.nn.row!=None:
                self.index_m(t)
                self.row_sum_list.append(None)
                self.rank_sum_list.append(None)
                self.row_probability.append(None)
                self.rank_probability.append(None)
        except AttributeError:
            self.running_flag=np.append(self.running_flag,np.array(1,dtype=np.int8))
        try:
            if self.nn.pr!=None:
                self.nn.pr.TD.append(np.array(0))
                self.nn.pr.index.append(None)
        except AttributeError:
            pass
        if self.threading!=None:
            self.pool_lock.append(self.threading.Lock())
            if self.row!=None:
                if self.d_index==0 or len(self.gradient_lock)<self.rank and len(self.gradient_lock[self.d_index-1])==self.row:
                    self.gradient_lock.append([])
                    self.d_index+=1
                self.gradient_lock[self.d_index-1].append(self.threading.Lock())
            elif self.PO==3 and len(self.gradient_lock)<self.max_lock:
                self.gradient_lock.append(self.threading.Lock())
        if self.PO==3:
            self.gradient_list.append(None)
        self.thread_counter+=1
        self.running_list.append(t)
        if self.memory_flag==True:
            self.grad_memory_list.append(0)
            self.pool_memory_list.append(0)
            self.episode_memory_list.append(0)
        if t>0:
            self.episode_list=np.append(self.episode_list,np.array(0,dtype=np.int8))
        self.finish_list.append(None)
        try:
            epsilon=self.epsilon[t]
        except:
            epsilon=None
        try:
            self.nn.ec.append(0)
        except AttributeError:
            pass
        try:
            self.nn.bc.append(0)
        except AttributeError:
            pass
        if self.PN==True:
            if self.PO==1 or self.PO==3:
                self.thread_lock[2].release()
            else:
                self.thread_lock[3].release()
        else:
            self.thread_lock[0].release()
        for k in range(episode_num):
            episode=[]
            s=self.nn.env(initial=True)
            if self.episode_step==None:
                while True:
                    next_s,r,done,_episode,index=self.env(s,epsilon,t)
                    self.reward[t]+=r
                    s=next_s
                    if type(self.done_pool[t])==np.ndarray:
                        self._train_(t)
                    if t in self.stop_list or t in self.stop_list_m:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].acquire()
                            else:
                                self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.thread_counter-=1
                        self.running_list.remove(t)
                        self.stop_list.remove(t)
                        self.stopped_list.append(t)
                        self.finish_list[t]=t
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].release()
                            else:
                                self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
                        del self.state_pool[t]
                        del self.action_pool[t]
                        del self.next_state_pool[t]
                        del self.reward_pool[t]
                        del self.done_pool[t]
                        return
                    if self.stop_flag==0:
                        del self.state_pool[t]
                        del self.action_pool[t]
                        del self.next_state_pool[t]
                        del self.reward_pool[t]
                        del self.done_pool[t]
                        return
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].acquire()
                            else:
                                self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.total_episode+=1
                        self.episode_list[t]+=1
                        self.loss_list.append(self.loss[t])
                        if self.trial_num!=None and len(self.reward_list)>=self.trial_num:
                            avg_reward=statistics.mean(self.reward_list[-self.trial_num:])
                            self.print_save(avg_reward)
                        else:
                            self.print_save()
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].release()
                            else:
                                self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
            else:
                for l in range(self.episode_step):
                    next_s,r,done,_episode,index=self.env(s,epsilon,t)
                    self.reward[t]+=r
                    s=next_s
                    if type(self.done_pool[t])==np.ndarray:
                        self._train_(t)
                    if t in self.stop_list or t in self.stop_list_m:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].acquire()
                            else:
                                self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.thread_counter-=1
                        self.running_list.remove(t)
                        self.stop_list.remove(t)
                        self.stopped_list.append(t)
                        self.finish_list[t]=t
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].release()
                            else:
                                self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
                        del self.state_pool[t]
                        del self.action_pool[t]
                        del self.next_state_pool[t]
                        del self.reward_pool[t]
                        del self.done_pool[t]
                        return
                    if self.stop_flag==0:
                        del self.state_pool[t]
                        del self.action_pool[t]
                        del self.next_state_pool[t]
                        del self.reward_pool[t]
                        del self.done_pool[t]
                        return
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].acquire()
                            else:
                                self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.total_episode+=1
                        self.episode_list[t]+=1
                        self.loss_list.append(self.loss[t])
                        if self.trial_num!=None and len(self.reward_list)>=self.trial_num:
                            avg_reward=statistics.mean(self.reward_list[-self.trial_num:])
                            self.print_save(avg_reward)
                        else:
                            self.print_save()
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].release()
                            else:
                                self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
                    if l==self.episode_step-1:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].acquire()
                            else:
                                self.thread_lock[3].acquire()
                        else:
                            self.thread_lock[0].acquire()
                        self.total_episode+=1
                        self.episode_list[t]+=1
                        self.loss_list.append(self.loss[t])
                        if self.trial_num!=None and len(self.reward_list)>=self.trial_num:
                            avg_reward=statistics.mean(self.reward_list[-self.trial_num:])
                            self.print_save(avg_reward)
                        else:
                            self.print_save()
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.thread_lock[2].release()
                            else:
                                self.thread_lock[3].release()
                        else:
                            self.thread_lock[0].release()
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.thread_lock[2].acquire()
                else:
                    self.thread_lock[3].acquire()
            else:
                self.thread_lock[0].acquire()
            self.reward_list.append(self.reward[t])
            self.reward[t]=0
            if self.save_episode==True:
                self.episode.append(episode)
                if self.max_episode_num!=None and len(self.episode)>=self.max_episode_num:
                    self.save_episode=False
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.thread_lock[2].release()
                else:
                    self.thread_lock[3].release()
            else:
                self.thread_lock[0].release()
        if self.PN==True:
            try:
                if self.nn.row!=None:
                    pass
            except AttributeError:
                self.running_flag[t+1]=0
            if self.PO==1 or self.PO==3:
                self.thread_lock[2].acquire()
            else:
                self.thread_lock[3].acquire()
            self.thread_counter-=1
            self.running_list.remove(t)
            if t not in self.finish_list:
                self.finish_list[t]=t
            if self.PO==1 or self.PO==3:
                self.thread_lock[2].release()
            else:
                self.thread_lock[3].release()
            try:
                del self.state_pool[t]
                del self.action_pool[t]
                del self.next_state_pool[t]
                del self.reward_pool[t]
                del self.done_pool[t]
            except:
                pass
        return
    
    
    def train_ol(self,t):
        self.exception_list.append(False)
        while True:
            if self.thread!=None:
                if self.save_flag==True:
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].acquire()
                    else:
                        self.thread_lock[2].acquire()
                    self.save()
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].release()
                    else:
                        self.thread_lock[2].release()
                    if self.stop_flag==2:
                        return
                if t in self.stop_list:
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].acquire()
                    else:
                        self.thread_lock[2].acquire()
                    self.stopped_list.append(t)
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].release()
                    else:
                        self.thread_lock[2].release()
                    return
                self.suspend_func(t)
                try:
                    data=self.nn.ol()
                except:
                    self.exception_list[t]=True
                    continue
                if data=='stop':
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].acquire()
                    else:
                        self.thread_lock[2].acquire()
                    self.stopped_list.append(t)
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].release()
                    else:
                        self.thread_lock[2].release()
                    return
                elif data=='suspend':
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].acquire()
                    else:
                        self.thread_lock[2].acquire()
                    self.suspended_list.append(t)
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].release()
                    else:
                        self.thread_lock[2].release() 
                    while True:
                        if t not in self.suspended_list:
                            break
                    continue
                try:
                    loss=self.opt_ol(data,t)
                except:
                    self.exception_list[t]=True
                    continue
                if self.stop_flag==0:
                    return
                if self.thread_lock!=None:
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].acquire()
                    else:
                        self.thread_lock[2].acquire()
                    loss=loss.numpy()
                    self.nn.train_loss_list.append(loss)
                    if len(self.nn.train_acc_list)==self.nn.max_length:
                        del self.nn.train_acc_list[0]
                    try:
                        self.nn.c+=1
                    except AttributeError:
                        pass
                    if self.PO==1 or self.PO==3:
                        self.thread_lock[1].release()
                    else:
                        self.thread_lock[2].release()
            else:
                if self.save_flag==True:
                    self.save()
                if self.stop_flag==2:
                    return
                self.suspend_func()
                try:
                    data=self.nn.ol()
                except:
                    self.exception_list[t]=True
                    continue
                if data=='stop':
                    self.stopped_list.append(t)
                    return
                elif data=='suspend':
                    self.suspended_list.append(t)
                    while True:
                        if t not in self.suspended_list:
                            break
                    continue
                try:
                    loss=self.opt_ol(data)
                except:
                    self.exception_list[t]=True
                    continue
                if self.stop_flag==0:
                    return
                loss=loss.numpy()
                self.nn.train_loss_list.append(loss)
                if len(self.nn.train_acc_list)==self.nn.max_length:
                    del self.nn.train_acc_list[0]
                try:
                    self.nn.c+=1
                except AttributeError:
                    pass
            self.exception_list[t]=False
        return
    
    
    def suspend_func(self,t):
        if t in self.suspend_list:
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.thread_lock[2].acquire()
                else:
                    self.thread_lock[3].acquire()
            else:
                self.thread_lock[0].acquire()
            self.suspended_list.append(t)
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.thread_lock[2].release()
                else:
                    self.thread_lock[3].release()
            else:
                self.thread_lock[0].release()
            while True:
                if t not in self.suspend_list:
                    if self.PN==True:
                        if self.PO==1 or self.PO==3:
                            self.thread_lock[2].acquire()
                        else:
                            self.thread_lock[3].acquire()
                    else:
                        self.thread_lock[0].acquire()
                    self.suspended_list.remove(t)
                    if self.PN==True:
                        if self.PO==1 or self.PO==3:
                            self.thread_lock[2].release()
                        else:
                            self.thread_lock[3].release()
                    else:
                        self.thread_lock[0].release()
                    break
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
                    self.save(self.total_episode,True)
                    self.save_flag=True
                    self.stop_flag=0
                    return True
        elif self.end():
            self.save(self.total_episode,True)
            self.save_flag=True
            self.stop_flag=0
            return True
        elif self.stop_flag==2:
            self.stop_flag=0
            return True
        return False
    
    
    def stop_func_(self,thread_lock):
        if self.stop==True and (self.stop_flag==1 or self.stop_flag==2):
            if self.stop_flag==0 or self.stop_func():
                thread_lock.release
                return True
    
    
    def stop_func_m(self,thread_lock,ln=None):
        if self.memory_t_value!=None and self.c_memory>self.memory_t_value:
            if self.memory_priority==False:
                if self.episode_list_copy==None:
                    self.episode_list_copy=self.episode_list.copy()
                index=np.argmax(self.episode_list_copy)
                self.stop_list_m.append(index)
                self.episode_list_copy[index]=0
                return False
            else:
                if self.PO==3:
                    self.grad_memory_list[ln]=0
                thread_lock.release()
                return True
        else:
            if self.memory_priority==False:
                self.stop_list_m.clear()
                self.episode_list_copy=None
            return False
    
    
    def stop_func_t_p(self,thread_lock,t,ln=None):
        if t in self.stop_list_m:
            self.pool_memory_list[t]=0
            if self.PO==3:
                self.grad_memory_list[ln]=0
            self.episode_list[t]=0
            thread_lock.release()
            return True
    
    
    def print_save(self,avg_reward=None):
        if self.muti_p!=None or self.muti_s!=None:
            if self.episode_num%10!=0:
                if self.muti_p!=None:
                    p=self.episode_num-self.episode_num%self.muti_p
                    p=int(p/self.muti_p)
                    if p==0:
                        p=1
                if self.muti_s!=None:
                    s=self.episode_num-self.episode_num%self.muti_s
                    s=int(s/self.muti_s)
                    if s==0:
                        s=1
            else:
                if self.muti_p!=None:
                    p=self.episode_num/(self.muti_p+1)
                    p=int(p)
                    if p==0:
                        p=1
                if self.muti_s!=None:
                    s=self.episode_num/(self.muti_s+1)
                    s=int(s)
                    if s==0:
                        s=1
            try:
                print('episode:{0}   loss:{1:.6f}'.format(self.total_episode,self.loss_list[-1]))
            except IndexError:
                pass
            if avg_reward!=None:
                print('episode:{0}   average reward:{1}'.format(self.total_episode,avg_reward))
            else:
                print('episode:{0}   reward:{1}'.format(self.total_episode,self.reward_list[-1]))
            print()
            if self.muti_s!=None and self.muti_save!=None and self.episode_num%s==0:
                if self.muti_save==1:
                    self.save(self.total_episode)
                else:
                    self.save(self.total_episode,False)
            self.episode_num+=1
        return
    
    
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
    
    
    def visualize_reward_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_list)),self.reward_list,'r-',label='reward')
        plt.plot(np.arange(len(self.loss_list)),self.loss_list,'b-',label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('reward and loss')
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
            self.file_list.append(['save-{0}.dat'])
            if len(self.file_list)>self.s+1:
                os.remove(self.file_list[0][0])
                del self.file_list[0]
            if self.save_episode==True:
                episode_file=open('episode-{0}.dat'.format(i),'wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
            if self.save_episode==True:
                self.file_list.append(['save-{0}.dat','episode-{0}.dat'])
                if len(self.file_list)>self.s+1:
                    os.remove(self.file_list[0][0])
                    os.remove(self.file_list[0][1])
                    del self.file_list[0]
            else:
                self.file_list.append(['save-{0}.dat'])
                if len(self.file_list)>self.s+1:
                    os.remove(self.file_list[0][0])
                    del self.file_list[0]
        try:
            if self.platform.DType!=None:
                try:
                    pickle.dump(self.nn,output_file)
                except AttributeError:
                    opt=self.nn.opt
                    self.nn.opt=None
                    pickle.dump(self.nn,output_file)
                    self.nn.opt=opt
        except AttributeError:
            pass
        try:
            pickle.dump(opt.get_config(),output_file)
        except:
            pickle.dump(None,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.sc,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.PN,output_file)
        pickle.dump(self.episode_memory_t_value,output_file)
        pickle.dump(self.memory_t_value,output_file)
        pickle.dump(self.max_episode_num,output_file)
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
        self.epsilon=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.sc=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.PN=pickle.load(input_file)
        self.episode_memory_t_value=pickle.load(input_file)
        self.memory_t_value=pickle.load(input_file)
        self.max_episode_num=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
