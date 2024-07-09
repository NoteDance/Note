import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Value,Array
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import os


class kernel:
    def __init__(self,nn=None,process=None):
        self.nn=nn
        self.nn.km=1
        if process!=None:
            self.reward=np.zeros(process,dtype='float32')
            self.sc=np.zeros(process,dtype='int32')
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.episode=None
        self.batch=None
        self.update_step=None
        self.trial_count=None
        self.process=process
        self.PO=3
        self.priority_flag=False
        self.max_opt=None
        self.stop=False
        self.opt_counter=None
        self.save_freq=1
        self.max_save_files=None
        self.path=None
    
    
    def init(self,manager):
        self.state_pool=manager.dict({})
        self.action_pool=manager.dict({})
        self.next_state_pool=manager.dict({})
        self.reward_pool=manager.dict({})
        self.done_pool=manager.dict({})
        self.reward=Array('f',self.reward)
        self.loss=np.zeros(self.process,dtype='float32')
        self.loss=Array('f',self.loss)
        self.sc=Array('i',self.sc)
        self.process_counter=Value('i',0)
        self.probability_list=manager.list([])
        self.running_flag_list=manager.list([])
        self.finish_list=manager.list([])
        self.running_flag=manager.list([0])
        self.reward_list=manager.list([])
        self.loss_list=manager.list([])
        self.episode_counter=Value('i',0)
        self.total_episode=Value('i',0)
        self.priority_p=Value('i',0)
        if self.priority_flag==True:
            self.opt_counter=Array('i',np.zeros(self.process,dtype='int32'))
        if self.nn is not None:
            self.nn.opt_counter=manager.list([tf.Variable(tf.zeros([self.process]))])  
        self._epoch_counter=manager.list([tf.Variable(0) for _ in range(self.process)])
        self.nn.ec=manager.list([0])
        self.ec=self.nn.ec[0]
        self._batch_counter=manager.list([tf.Variable(0) for _ in range(self.process)])
        self.nn.bc=manager.list([0])
        self.bc=self.nn.bc[0]
        self.episode_=Value('i',self.total_episode.value)
        self.stop_flag=Value('b',False)
        self.save_flag=Value('b',False)
        self.path_list=manager.list([])
        self.param=manager.dict()
        self.param[7]=self.nn.param
        return
    
    
    def init_online(self,manager):
        self.nn.train_loss_list=manager.list([])
        self.nn.counter=manager.list([])
        self.nn.exception_list=manager.list([])
        self.param=manager.dict()
        self.param[7]=self.nn.param
        return
    
    
    def action_vec(self):
        self.action_one=np.ones(self.action_count,dtype=np.int8)
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
        if epsilon!=None:
            self.epsilon=np.ones(self.process)*epsilon
            self.action_vec()
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_step!=None:
            self.update_step=update_step
        if trial_count!=None:
            self.trial_count=trial_count
        if criterion!=None:
            self.criterion=criterion
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon):
        action_prob=self.action_one*epsilon/len(self.action_one)
        best_a=np.argmax(self.nn.nn.fp(s))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        pool_lock[index].acquire()
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None:
                self.state_pool[index]=s
                if type(a)==int:
                    a=np.array(a)
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
                        a=np.array(a)
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                    else:
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                except Exception:
                    pass
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size:
                self.state_pool[index]=self.state_pool[index][1:]
                self.action_pool[index]=self.action_pool[index][1:]
                self.next_state_pool[index]=self.next_state_pool[index][1:]
                self.reward_pool[index]=self.reward_pool[index][1:]
                self.done_pool[index]=self.done_pool[index][1:]
        except Exception:
            pool_lock[index].release()
            return
        pool_lock[index].release()
        return
    
    
    def get_index(self,p,lock):
        while len(self.running_flag_list)<p:
            pass
        if len(self.running_flag_list)==p:
            if self.PO==1 or self.PO==2:
                lock[2].acquire()
            elif self.PO==3:
                lock[0].acquire()
            self.running_flag_list.append(self.running_flag[1:].copy())
            if self.PO==1 or self.PO==2:
                lock[2].release()
            elif self.PO==3:
                lock[0].release()
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value:
            self.running_flag_list[p]=self.running_flag[1:].copy()
        while len(self.probability_list)<p:
            pass
        if len(self.probability_list)==p:
            if self.PO==1 or self.PO==2:
                lock[2].acquire()
            elif self.PO==3:
                lock[0].acquire()
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]))
            if self.PO==1 or self.PO==2:
                lock[2].release()
            elif self.PO==3:
                lock[0].release()
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])
        while True:
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p])
            if index in self.finish_list:
                continue
            else:
                break
        return index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):
        if hasattr(self.nn,'nn'):
            s=np.expand_dims(s,axis=0)
            if epsilon==None:
                action_prob=self.nn.nn.fp(s)
            else:
                action_prob=self.epsilon_greedy_policy(s,epsilon)
            a=np.random.choice(self.action_count,p=action_prob)
        else:
            if hasattr(self.nn,'action'):
                s=np.expand_dims(s,axis=0)
                a=self.nn.action(s).numpy()
            else:
                s=np.expand_dims(s,axis=0)
                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
        next_s,r,done=self.nn.env(a,p)
        index=self.get_index(p,lock)
        if type(self.nn.param[0])!=list:
            next_s=np.array(next_s,self.nn.param[0].dtype.name)
            r=np.array(r,self.nn.param[0].dtype.name)
            done=np.array(done,self.nn.param[0].dtype.name)
        else:
            next_s=np.array(next_s,self.nn.param[0][0].dtype.name)
            r=np.array(r,self.nn.param[0][0].dtype.name)
            done=np.array(done,self.nn.param[0][0].dtype.name)
        self.pool(s,a,next_s,r,done,pool_lock,index)
        return next_s,r,done,index
    
    
    def end(self):
        if self.trial_count!=None:
            if len(self.reward_list)>=self.trial_count:
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                if self.criterion!=None and avg_reward>=self.criterion:
                    return True
        return False
    
    
    @tf.function(jit_compile=True)
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock=None):
        with tf.GradientTape(persistent=True) as tape:
            try:
                try:
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                except Exception:
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)
            except Exception as e:
                raise e
        if self.PO==1:
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if self.stop_flag.value==True:
                        return None,None
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return None,None
            if hasattr(self.nn,'gradient'):
                gradient=self.nn.gradient(tape,loss)
            else:
                if hasattr(self.nn,'nn'):
                    gradient=tape.gradient(loss,self.nn.param)
                else:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            try:
                if hasattr(self.nn,'attenuate'):
                    try:
                        gradient=self.nn.attenuate(gradient,p)
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,p)
                        critic_gradient=self.nn.attenuate(critic_gradient,p)
            except Exception as e:
                raise e
            try:
                try:
                    param=self.nn.opt(gradient,p)
                except Exception:
                    param=self.nn.opt(gradient)
            except Exception as e:
                raise e
            lock[0].release()
        elif self.PO==2:
            g_lock.acquire()
            if self.stop_func_(g_lock):
                return None,None
            if hasattr(self.nn,'gradient'):
                gradient=self.nn.gradient(tape,loss)
            else:
                if hasattr(self.nn,'nn'):
                    gradient=tape.gradient(loss,self.nn.param)
                else:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            g_lock.release()
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if self.stop_flag.value==True:
                        return None,None
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return None,None
            try:
                if hasattr(self.nn,'attenuate'):
                    try:
                        gradient=self.nn.attenuate(gradient,p)
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,p)
                        critic_gradient=self.nn.attenuate(critic_gradient,p)
            except Exception as e:
                raise e
            try:
                try:
                    param=self.nn.opt(gradient,p)
                except Exception:
                    param=self.nn.opt(gradient)
            except Exception as e:
                raise e
            lock[0].release()
        elif self.PO==3:
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if self.stop_flag.value==True:
                        return None,None
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            if self.stop_func_():
                return None,None
            if hasattr(self.nn,'gradient'):
                gradient=self.nn.gradient(tape,loss)
            else:
                if hasattr(self.nn,'nn'):
                    gradient=tape.gradient(loss,self.nn.param)
                else:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            try:
                if hasattr(self.nn,'attenuate'):
                    try:
                        gradient=self.nn.attenuate(gradient,p)
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,p)
                        critic_gradient=self.nn.attenuate(critic_gradient,p)
            except Exception as e:
                raise e
            try:
                try:
                    param=self.nn.opt(gradient,p)
                except Exception:
                    param=self.nn.opt(gradient)
            except Exception as e:
                raise e
        return loss,param
    
    
    def update_nn_param(self,param=None):
        if param==None:
            parameter_flat=nest.flatten(self.nn.param)
            parameter7_flat=nest.flatten(self.param[7])
        else:
            parameter_flat=nest.flatten(self.nn.param)
            parameter7_flat=nest.flatten(param)
        for i in range(len(parameter_flat)):
            if param==None:
                state_ops.assign(parameter_flat[i],parameter7_flat[i])
            else:
                state_ops.assign(parameter_flat[i],parameter7_flat[i])
        self.nn.param=nest.pack_sequence_as(self.nn.param,parameter_flat)
        self.param[7]=nest.pack_sequence_as(self.param[7],parameter7_flat)
        return
    
    
    def _train(self,p,j,batches,length,lock,g_lock):
        if j==batches-1:
            index1=batches*self.batch
            index2=self.batch-(length-batches*self.batch)
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0)
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0)
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0)
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0)
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0)
            if self.PO==2:
                if type(g_lock)!=list:
                    pass
                elif len(g_lock)==self.process:
                    ln=p
                    g_lock=g_lock[ln]
                else:
                    ln=int(np.random.choice(len(g_lock)))
                    g_lock=g_lock[ln]
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock)
            else:
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock)
            if self.stop_flag.value==True:
                return
            self.param[7]=param
            self.loss[p]+=loss
            self.nn.bc[0]=sum(self._batch_counter)+self.bc
            _batch_counter=self._batch_counter[p]
            _batch_counter.assign_add(1)
            self._batch_counter[p]=_batch_counter
        else:
            index1=j*self.batch
            index2=(j+1)*self.batch
            state_batch=self.state_pool[p][index1:index2]
            action_batch=self.action_pool[p][index1:index2]
            next_state_batch=self.next_state_pool[p][index1:index2]
            reward_batch=self.reward_pool[p][index1:index2]
            done_batch=self.done_pool[p][index1:index2]
            if self.PO==2:
                if type(g_lock)!=list:
                    pass
                elif len(g_lock)==self.process:
                    ln=p
                    g_lock=g_lock[ln]
                else:
                    ln=int(np.random.choice(len(g_lock)))
                    g_lock=g_lock[ln]
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock)
            else:
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock)
            if self.stop_flag.value==True:
                return
            self.param[7]=param
            self.loss[p]+=loss
            self.nn.bc[0]=sum(self._batch_counter)+self.bc
            _batch_counter=self._batch_counter[p]
            _batch_counter.assign_add(1)
            self._batch_counter[p]=_batch_counter
        return
    
    
    def train_(self,p,lock,g_lock):
        if len(self.done_pool[p])<self.batch:
            return
        else:
            self.loss[p]=0
            length=len(self.done_pool[p])
            batches=int((length-length%self.batch)/self.batch)
            if length%self.batch!=0:
                batches+=1
            for j in range(batches):
                if self.priority_flag==True:
                    self.priority_p.value=np.argmax(self.opt_counter)
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt:
                        self.priority_p.value=int(self.priority_p.value)
                    elif self.max_opt==None:
                        self.priority_p.value=int(self.priority_p.value)
                    else:
                        self.priority_p.value=-1
                if self.priority_flag==True:
                    self.opt_counter[p]=0
                if hasattr(self.nn,'attenuate'):
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter.scatter_update(tf.IndexedSlices(0,p))
                    self.nn.opt_counter[0]=opt_counter
                self._train(p,j,batches,length,lock,g_lock)
                if self.stop_flag.value==True:
                    return
                if self.priority_flag==True:
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')
                    opt_counter+=1
                if hasattr(self.nn,'attenuate'):
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter.assign(opt_counter+1)
                    self.nn.opt_counter[0]=opt_counter
            if self.PO==1 or self.PO==2:
                lock[1].acquire()
            if self.update_step!=None:
                if self.sc[p]%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
            if self.PO==1 or self.PO==2:
                lock[1].release()
            self.loss[p]=self.loss[p]/batches
        self.sc[p]+=1
        self.nn.ec[0]=sum(self._epoch_counter)+self.ec
        _epoch_counter=self._epoch_counter[p]
        _epoch_counter.assign_add(1)
        self._epoch_counter[p]=_epoch_counter
        return
    
    
    def train(self,p,lock,pool_lock,g_lock=None):
        if self.PO==1 or self.PO==2:
            lock[1].acquire()
        elif self.PO==3:
            lock[1].acquire()
        self.state_pool[p]=None
        self.action_pool[p]=None
        self.next_state_pool[p]=None
        self.reward_pool[p]=None
        self.done_pool[p]=None
        self.running_flag.append(1)
        self.process_counter.value+=1
        self.finish_list.append(None)
        if self.PO==1 or self.PO==2:
            lock[1].release()
        elif self.PO==3:
            lock[1].release()
        try:
            epsilon=self.epsilon[p]
        except Exception:
            epsilon=None
        while True:
            if self.stop_flag.value==True:
                break
            if self.episode!=None and self.episode_counter.value>=self.episode:
                break
            s=self.nn.env(p=p,initial=True)
            if type(self.nn.param[0])!=list:
                s=np.array(s,self.nn.param[0].dtype.name)
            else:
                s=np.array(s,self.nn.param[0][0].dtype.name)
            if self.episode_step==None:
                while True:
                    if self.episode!=None and self.episode_counter.value>=self.episode:
                        break
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)
                    self.reward[p]+=r
                    s=next_s
                    if type(self.done_pool[p])==np.ndarray:
                        self.train_(p,lock,g_lock)
                        if self.stop_flag.value==True:
                            break
                    if done:
                        if self.PO==1 or self.PO==2:
                            lock[1].acquire()
                        elif len(lock)==4:
                            lock[3].acquire()
                        self.episode_counter.value+=1
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if self.PO==1 or self.PO==2:
                            lock[1].release()
                        elif len(lock)==4:
                            lock[3].release()
                        break
            else:
                for l in range(self.episode_step):
                    if self.episode!=None and self.episode_counter.value>=self.episode:
                        break
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)
                    self.reward[p]+=r
                    s=next_s
                    if type(self.done_pool[p])==np.ndarray:
                        self.train_(p,lock,g_lock)
                        if self.stop_flag.value==True:
                            break
                    if done:
                        if self.PO==1 or self.PO==2:
                            lock[1].acquire()
                        elif len(lock)==4:
                            lock[3].acquire()
                        self.episode_counter.value+=1
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if self.PO==1 or self.PO==2:
                            lock[1].release()
                        elif len(lock)==4:
                            lock[3].release()
                        break
                    if l==self.episode_step-1:
                        if self.PO==1 or self.PO==2:
                            lock[1].acquire()
                        elif len(lock)==4:
                            lock[3].acquire()
                        self.episode_counter.value+=1
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if self.PO==1 or self.PO==2:
                            lock[1].release()
                        elif len(lock)==4:
                            lock[3].release()
            if self.PO==1 or self.PO==2:
                lock[1].acquire()
            elif len(lock)==3 or len(lock)==4:
                lock[2].acquire()
            self.save_()
            self.reward_list.append(self.reward[p])
            self.reward[p]=0
            if self.PO==1 or self.PO==2:
                lock[1].release()
            elif len(lock)==3 or len(lock)==4:
                lock[2].release()
        self.running_flag[p+1]=0
        if p not in self.finish_list:
            self.finish_list[p]=p
        if self.PO==1 or self.PO==2:
            lock[1].acquire()
        elif self.PO==3:
            lock[1].acquire()
        self.process_counter.value-=1
        if self.PO==1 or self.PO==2:
            lock[1].release()
        elif self.PO==3:
            lock[1].release()
        del self.state_pool[p]
        del self.action_pool[p]
        del self.next_state_pool[p]
        del self.reward_pool[p]
        del self.done_pool[p]
        return
    
    
    def train_online(self,p,lock=None,g_lock=None):
        if hasattr(self.nn,'counter'):
            self.nn.counter.append(0)
        while True:
            if hasattr(self.nn,'save'):
                self.nn.save(self.save,p)
            if hasattr(self.nn,'stop_flag'):
                if self.nn.stop_flag==True:
                    return
            if hasattr(self.nn,'stop_func'):
                if self.nn.stop_func(p):
                    return
            if hasattr(self.nn,'suspend_func'):
                self.nn.suspend_func(p)
            try:
                data=self.nn.online(p)
            except Exception as e:
                self.nn.exception_list[p]=e
            if data=='stop':
                return
            elif data=='suspend':
                self.nn.suspend_func(p)
            try:
                if self.PO==2:
                    if type(g_lock)!=list:
                        pass
                    elif len(g_lock)==self.process:
                        ln=p
                        g_lock=g_lock[ln]
                    else:
                        ln=int(np.random.choice(len(g_lock)))
                        g_lock=g_lock[ln]
                loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock,g_lock)
                self.param[7]=param
            except Exception as e:
                if self.PO==1:
                    if lock[0].acquire(False):
                        lock[0].release()
                elif self.PO==2:
                    if g_lock.acquire(False):
                        g_lock.release()
                    if lock[0].acquire(False):
                        lock[0].release()
                self.nn.exception_list[p]=e
            loss=loss.numpy()
            if len(self.nn.train_loss_list)==self.nn.max_length:
                del self.nn.train_loss_list[0]
            self.nn.train_loss_list.append(loss)
            try:
                if hasattr(self.nn,'counter'):
                    count=self.nn.counter[p]
                    count+=1
                    self.nn.counter[p]=count
            except IndexError:
                self.nn.counter.append(0)
                count=self.nn.counter[p]
                count+=1
                self.nn.counter[p]=count
        return
    
    
    def stop_func(self):
        if self.end():
            self.save(self.total_episode)
            self.save_flag.value=True
            self.stop_flag.value=True
            return True
        return False
    
    
    def stop_func_(self,lock=None):
        if self.stop==True:
            if self.stop_flag.value==True or self.stop_func():
                if self.PO!=3:
                    lock.release
                return True
        return False
    
    
    def save_(self):
        if self.path!=None and self.episode_.value%self.save_freq==0:
            self.save()
        self.episode_.value+=1
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
    
    
    def save_param(self,path):
        parameter_file=open(path,'wb')
        pickle.dump(self.param[7],parameter_file)
        parameter_file.close()
        return
    
    
    def restore_param(self,path):
        parameter_file=open(path,'rb')
        param=pickle.load(parameter_file)
        param_flat=nest.flatten(param)
        param_flat_=nest.flatten(self.nn.param)
        for i in range(len(param_flat)):
            state_ops.assign(param_flat_[i],param_flat[i])
        self.nn.param=nest.pack_sequence_as(self.nn.param,param_flat_)
        parameter_file.close()
        return
    
    
    def save(self):
        if self.save_flag.value==True:
            return
        if self.max_save_files==None:
            output_file=open(self.path,'wb')
        else:
            path=self.path.replace(self.path[self.path.find('.'):],'-{0}.dat'.format(self.total_episode.value))
            output_file=open(path,'wb')
            self.file_list.append([path])
            if len(self.path_list)>self.max_save_files:
                os.remove(self.path_list[0][0])
                del self.path_list[0]
        self.update_nn_param()
        self.nn.opt_counter=None
        self.nn.ec=self.nn.ec[0]
        self.nn.bc=self.nn.bc[0]
        self._epoch_counter=list(self._epoch_counter)
        self._batch_counter=list(self._batch_counter)
        self.nn.optimizer.convert_to_list()
        pickle.dump(self.nn,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(np.array(self.sc,dtype='int32'),output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(list(self.reward_list),output_file)
        pickle.dump(list(self.loss_list),output_file)
        pickle.dump(self.total_episode.value,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,manager):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        self.convert_to_shared_list(manager)
        self.nn.km=1
        self.nn.opt_counter=manager.list([tf.Variable(tf.zeros([self.process]))])
        self.ec=self.nn.ec
        self.bc=self.nn.bc
        self.param[7]=self.nn.param
        self.epsilon=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.sc=pickle.load(input_file)
        self.sc=Array('i',self.sc)
        self.update_step=pickle.load(input_file)
        self.reward_list[:]=pickle.load(input_file)
        self.loss_list[:]=pickle.load(input_file)
        self.total_episode.value=pickle.load(input_file)
        self.episode_.value=self.total_episode.value
        input_file.close()
        return
