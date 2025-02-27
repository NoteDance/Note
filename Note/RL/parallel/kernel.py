import tensorflow as tf
from Note.RL import rl
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
        self.state_pool=None
        self.step_counter=None
        self.pool_size=None
        self.episode=None
        self.batch=None
        self.update_steps=None
        self.trial_count=None
        self.process=process
        self.PO=3
        self.priority_flag=False
        self.max_opt=None
        self.stop=False
        self.opt_counter=None
        self.path=None
        self.save_data=True
        self.save_freq=1
        self.max_save_files=None
        self.save_best_only=False
        self.save_param_only=False
    
    
    def init(self,manager):
        if self.state_pool==None or not self.save_data:
            self.state_pool=manager.dict()
            self.action_pool=manager.dict()
            self.next_state_pool=manager.dict()
            self.reward_pool=manager.dict()
            self.done_pool=manager.dict()
        else:
            self.state_pool=manager.dict(self.state_pool)
            self.action_pool=manager.dict(self.action_pool)
            self.next_state_pool=manager.dict(self.next_state_pool)
            self.reward_pool=manager.dict(self.reward_pool)
            self.done_pool=manager.dict(self.done_pool)
        self.reward=Array('f',np.zeros(self.process,dtype='float32'))
        self.loss=np.zeros(self.process,dtype='float32')
        self.loss=Array('f',self.loss)
        if self.step_counter is None:
            self.step_counter=Array('i',np.zeros(self.process,dtype='int32'))
        elif self.process>len(self.step_counter):
            self.step_counter=Array('i',np.concatenate((self.step_counter,np.zeros(self.process-len(self.step_counter),dtype='int32'))))
        else:
            self.step_counter=Array('i',self.step_counter)
        self.process_counter=Value('i',0)
        self.finish_list=manager.list()
        self.reward_list=manager.list()
        self.loss_list=manager.list()
        self.episode_counter=Value('i',0)
        self.total_episode=Value('i',0)
        self.priority_p=Value('i',0)
        self.inverse_len=manager.list([0 for _ in range(self.process)])
        if self.PR:
            self.nn.pr.TD=manager.list([[self.nn.initial_TD] for _ in range(self.process)])
        if self.priority_flag==True:
            self.opt_counter=Array('i',np.zeros(self.process,dtype='int32'))
        if self.nn is not None:
            self.nn.opt_counter=manager.list([tf.Variable(tf.zeros([self.process]))])  
        self._episode_counter=manager.list([tf.Variable(0) for _ in range(self.process)])
        self.nn.ec=manager.list([0])
        self.ec=self.nn.ec[0]
        self._batch_counter=manager.list([tf.Variable(0) for _ in range(self.process)])
        self.nn.bc=manager.list([0])
        self.bc=self.nn.bc[0]
        self.episode_=Value('i',self.total_episode.value)
        self.stop_flag=Value('b',False)
        self.save_flag=Value('b',False)
        self.path_list=manager.list()
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
    
    
    def set_up(self,policy=None,noise=None,pool_size=None,batch=None,update_steps=None,trial_count=None,criterion=None,PPO=None,HER=None,MARL=None,PR=None,IRL=None):
        if policy!=None:
            self.policy=policy
        if noise!=None:
            self.noise=noise
            self.nn.noise=True
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_steps!=None:
            self.update_steps=update_steps
        if trial_count!=None:
            self.trial_count=trial_count
        if criterion!=None:
            self.criterion=criterion
        if self.PPO!=None:
            self.PPO=PPO
        if self.HER!=None:
            self.HER=HER
        if self.MARL!=None:
            self.MARL=MARL
        if self.PR!=None:
            self.PR=PR
        if self.IRL!=None:
            self.IRL=IRL
        return
    
    
    def run_agent(self, max_steps, seed=None):
        state_history = []

        steps = 0
        reward_ = 0
        if seed==None:
            state = self.nn.genv.reset()
        else:
            state = self.nn.genv.reset(seed=seed)
        for step in range(max_steps):
            if not hasattr(self, 'noise'):
                action = np.argmax(self.nn.nn(state))
            else:
                action = self.nn.actor(state).numpy()
            next_state, reward, done, _ = self.nn.genv.step(action)
            state_history.append(state)
            steps+=1
            reward_+=reward
            if done:
                break
            state = next_state
        
        return state_history,reward_,steps
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        if self.HER!=True or self.PR!=True:
            pool_lock[index].acquire()
        try:
            if self.state_pool[index] is None:
                self.state_pool[index]=s
                self.action_pool[index]=np.expand_dims(a,axis=0)
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0)
                self.reward_pool[index]=np.expand_dims(r,axis=0)
                self.done_pool[index]=np.expand_dims(done,axis=0)
            else:
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)
                    self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
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
            if self.HER!=True or self.PR!=True:
                pool_lock[index].release()
            return
        if self.HER!=True or self.PR!=True:
            pool_lock[index].release()
        return
    
    
    def get_index(self,p):
        inverse_len=tf.constant(self.inverse_len)
        total_inverse=tf.reduce_sum(inverse_len)
        prob=inverse_len/total_inverse
        while True:
            index=np.random.choice(self.processes,p=prob.numpy())
            if index in self.finish_list:
                self.inverse_len[index]=0
                continue
            else:
                self.inverse_len[index]=1/(len(self.state_pool[index])+1)
                break
        return index
    
    
    @tf.function(jit_compile=True)
    def forward(self,s,i):
        if self.MARL!=True:
            if hasattr(self.nn,'nn'):
                output=self.nn.nn(s)
            else:
                output=self.nn.actor(s)
        else:
            if hasattr(self.nn,'nn'):
                output=self.nn.nn(s,i)
            else:
                output=self.nn.actor(s,i)
        return output
    
    
    def select_action(self,s,i=None):
        output=self.forward(s,i)
        if hasattr(self.nn,'nn'):
            if self.IRL!=True:
                output=output.numpy()
            else:
                output=output[1].numpy()
            output=np.squeeze(output, axis=0)
            if isinstance(self.policy, rl.SoftmaxPolicy):
                a=self.policy.select_action(len(output), output)
            elif isinstance(self.policy, rl.EpsGreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.AdaptiveEpsGreedyPolicy):
                a=self.policy.select_action(output, np.sum(self.step_counter))
            elif isinstance(self.policy, rl.GreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.MaxBoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannGumbelQPolicy):
                a=self.policy.select_action(output, np.sum(self.step_counter))
        else:
            if self.IRL!=True:
                a=(output+self.noise.sample()).numpy()
            else:
                a=(output[1]+self.noise.sample()).numpy()
        if self.IRL!=True:
            return a
        else:
            return [output[0],a]
    
    
    def store(self,s,p,lock,pool_lock):
        s=np.expand_dims(s,axis=0)
        if self.MARL!=True:
            a=self.select_action(s)
        else:
            a=[]
            for i in len(s[0]):
                s=np.expand_dims(s[0][i],axis=0)
                a.append([self.select_action(s,i)])
            a=np.array(a)
        next_s,r,done=self.nn.env(a,p)
        if self.HER!=True or self.PR!=True:
            if type(self.state_pool[p])!=np.ndarray and self.state_pool[p]==None:
                index=p
                self.inverse_len[index]=1
            else:
                index=self.get_index(p)
        else:
            index=p
        next_s=np.array(next_s)
        r=np.array(r)
        done=np.array(done)
        self.pool(s,a,next_s,r,done,pool_lock,index)
        return next_s,r,done
    
    
    def data_func(self,p):
        s = []
        a = []
        next_s = []
        r = []
        d = []
        for _ in range(self.batch):
            step_state = np.random.randint(0, len(self.state_pool[p])-1)
            step_goal = np.random.randint(step_state+1, step_state+np.argmax(self.done_pool[p][step_state+1:])+2)
            state = self.state_pool[p][step_state]
            next_state = self.next_state_pool[p][step_state]
            action = self.action_pool[p][step_state]
            goal = self.state_pool[p][step_goal]
            reward, done = self.nn.reward_done_func(next_state, goal)
            state = np.hstack((state, goal))
            next_state = np.hstack((next_state, goal))
            s.append(state)
            a.append(action)
            next_s.append(next_state)
            r.append(reward)
            d.append(done)
        s = np.array(s)
        a = np.array(a)
        next_s = np.array(next_s)
        r = np.array(r)
        d = np.array(d)
        return s,a,next_s,r,d
    
    
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
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            except Exception:
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)
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
                gradient=self.nn.gradient(tape,loss,self.nn.param)
            else:
                if hasattr(self.nn,'nn'):
                    gradient=tape.gradient(loss,self.nn.param)
                else:
                    actor_gradient=tape.gradient(loss,self.nn.param[0])
                    critic_gradient=tape.gradient(loss,self.nn.param[1])
                    gradient=[actor_gradient,critic_gradient]
            if hasattr(self.nn,'attenuate'):
                if hasattr(self.nn,'nn'):
                    gradient=self.nn.attenuate(gradient,p)
                else:
                    actor_gradient=self.nn.attenuate(actor_gradient,p)
                    critic_gradient=self.nn.attenuate(critic_gradient,p)
                    gradient=[actor_gradient,critic_gradient]
            try:
                param=self.nn.opt(gradient,p)
            except Exception:
                param=self.nn.opt(gradient)
            lock[0].release()
        elif self.PO==2:
            g_lock.acquire()
            if self.stop_func_(g_lock):
                return None,None
            if hasattr(self.nn,'gradient'):
                gradient=self.nn.gradient(tape,loss,self.nn.param)
            else:
                if hasattr(self.nn,'nn'):
                    gradient=tape.gradient(loss,self.nn.param)
                else:
                    actor_gradient=tape.gradient(loss,self.nn.param[0])
                    critic_gradient=tape.gradient(loss,self.nn.param[1])
                    gradient=[actor_gradient,critic_gradient]
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
            if hasattr(self.nn,'attenuate'):
                if hasattr(self.nn,'nn'):
                    gradient=self.nn.attenuate(gradient,p)
                else:
                    actor_gradient=self.nn.attenuate(actor_gradient,p)
                    critic_gradient=self.nn.attenuate(critic_gradient,p)
                    gradient=[actor_gradient,critic_gradient]
            try:
                param=self.nn.opt(gradient,p)
            except Exception:
                param=self.nn.opt(gradient)
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
                gradient=self.nn.gradient(tape,loss,self.nn.param)
            else:
                if hasattr(self.nn,'nn'):
                    gradient=tape.gradient(loss,self.nn.param)
                else:
                    actor_gradient=tape.gradient(loss,self.nn.param[0])
                    critic_gradient=tape.gradient(loss,self.nn.param[1])
                    gradient=[actor_gradient,critic_gradient]
            if hasattr(self.nn,'attenuate'):
                if hasattr(self.nn,'nn'):
                    gradient=self.nn.attenuate(gradient,p)
                else:
                    actor_gradient=self.nn.attenuate(actor_gradient,p)
                    critic_gradient=self.nn.attenuate(critic_gradient,p)
                    gradient=[actor_gradient,critic_gradient]
            try:
                param=self.nn.opt(gradient,p)
            except Exception:
                param=self.nn.opt(gradient)
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
        return
    
    
    def _train(self,p,j,batches,length,lock,g_lock):
        if j==batches-1:
            if self.PR:
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool[p],self.action_pool[p],self.next_state_pool[p],self.reward_pool[p],self.done_pool[p],self.batch,p)
            elif self.HER:
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func(p)
            else:
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
            if self.PR:
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool[p],self.action_pool[p],self.next_state_pool[p],self.reward_pool[p],self.done_pool[p],self.batch,p)
            elif self.HER:
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func(p)
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
            self.step_counter[p]+=1
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
            if self.update_steps!=None:
                if self.step_counter[p]%self.update_steps==0:
                    self.nn.update_param()
                    if self.PPO:
                        self.state_pool[p]=None
                        self.action_pool[p]=None
                        self.next_state_pool[p]=None
                        self.reward_pool[p]=None
                        self.done_pool[p]=None
            else:
                self.nn.update_param()
            if self.PO==1 or self.PO==2:
                lock[1].release()
            self.loss[p]=self.loss[p]/batches
        self.nn.ec[0]=sum(self._episode_counter)+self.ec
        _episode_counter=self._episode_counter[p]
        _episode_counter.assign_add(1)
        self._episode_counter[p]=_episode_counter
        return
    
    
    def train(self,p,lock,pool_lock,g_lock=None):
        if self.PO==1 or self.PO==2:
            lock[1].acquire()
        elif self.PO==3:
            lock[0].acquire()
        if not self.save_data:
            self.state_pool[p]=None
            self.action_pool[p]=None
            self.next_state_pool[p]=None
            self.reward_pool[p]=None
            self.done_pool[p]=None
        self.process_counter.value+=1
        self.finish_list.append(None)
        if self.PO==1 or self.PO==2:
            lock[1].release()
        elif self.PO==3:
            lock[0].release()
        while True:
            if self.stop_flag.value==True:
                break
            if self.episode!=None and self.episode_counter.value>=self.episode:
                break
            s=self.nn.env(p=p,initial=True)
            while True:
                if self.episode!=None and self.episode_counter.value>=self.episode:
                    break
                next_s,r,done=self.store(s,p,lock,pool_lock)
                self.reward[p]+=r
                s=next_s
                if self.PR:
                    self.nn.pr.TD[p]=np.append(self.nn.pr.TD[p],self.nn.initial_TD)
                    if len(self.state_pool[p])>self.pool_size:
                        self.nn.pr.TD[p]=self.nn.pr.TD[p][1:]
                    self.nn.pr.TD[p]=tf.Variable(self.nn.pr.TD[p])
                if type(self.done_pool[p])==np.ndarray:
                    self.train_(p,lock,g_lock)
                    if self.stop_flag.value==True:
                        break
                if done:
                    if self.PO==1 or self.PO==2:
                        lock[1].acquire()
                    elif self.PO==3:
                        lock[0].acquire()
                    self.reward_list.append(self.reward[p])
                    if len(self.reward_list)>self.trial_count:
                        del self.reward_list[0]
                    self.reward[p]=0
                    self.episode_counter.value+=1
                    self.total_episode.value+=1
                    self.loss_list.append(self.loss[p])
                    if self.PO==1 or self.PO==2:
                        lock[1].release()
                    elif self.PO==3:
                        lock[0].release()
                    break
            if self.PO==1 or self.PO==2:
                lock[1].acquire()
            elif self.PO==3:
                lock[0].acquire()
            if self.save_param_only==False:
                self.save_param_()
            else:
                self.save_()
            if self.PO==1 or self.PO==2:
                lock[1].release()
            elif self.PO==3:
                lock[0].release()
        self.inverse_len[p]=0
        if p not in self.finish_list:
            self.finish_list[p]=p
        if self.PO==1 or self.PO==2:
            lock[1].acquire()
        elif self.PO==3:
            lock[0].acquire()
        self.process_counter.value-=1
        if self.PO==1 or self.PO==2:
            lock[1].release()
        elif self.PO==3:
            lock[0].release()
        if not self.save_data:
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
                self.nn.save(self._save,p)
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
            self._save()
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
            self._save()
        self.episode_.value+=1
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def visualize_reward_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list,'r-',label='reward')
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list,'b-',label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('reward and loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        return
    
    
    def save_param_(self):
        if self.save_flag.value==True:
            return
        if self.save_best_only==False:
            if self.max_save_files==None:
                parameter_file=open(self.path,'wb')
            else:
                path=self.path.replace(self.path[self.path.find('.'):],'-{0}.dat'.format(self.total_episode.value))
                parameter_file=open(path,'wb')
                self.file_list.append([path])
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0][0])
                    del self.path_list[0]
            pickle.dump(self.param[7],parameter_file)
            parameter_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save_param(self.path)
                        self.avg_reward=avg_reward
        return
    
    
    def save_param(self,path):
        if self.max_save_files==None:
            parameter_file=open(path,'wb')
        else:
            path=self.path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_episode.value))
            parameter_file=open(path,'wb')
            self.file_list.append([path])
            if len(self.path_list)>self.max_save_files:
                os.remove(self.path_list[0][0])
                del self.path_list[0]
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
    
    
    def _save(self):
        if self.save_flag.value==True:
            return
        if self.save_best_only==False:
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
            self._episode_counter=list(self._episode_counter)
            self._batch_counter=list(self._batch_counter)
            self.nn.optimizer.convert_to_list()
            pickle.dump(self.nn,output_file)
            pickle.dump(self.state_pool,output_file)
            pickle.dump(self.action_pool,output_file)
            pickle.dump(self.next_state_pool,output_file)
            pickle.dump(self.reward_pool,output_file)
            pickle.dump(self.done_pool,output_file)
            pickle.dump(self.policy,output_file)
            pickle.dump(self.noise,output_file)
            pickle.dump(self.pool_size,output_file)
            pickle.dump(self.batch,output_file)
            pickle.dump(np.array(self.step_counter,dtype='int32'),output_file)
            pickle.dump(self.update_steps,output_file)
            pickle.dump(self.trial_count,output_file)
            pickle.dump(self.criterion,output_file)
            pickle.dump(self.PPO,output_file)
            pickle.dump(self.HER,output_file)
            pickle.dump(self.MARL,output_file)
            pickle.dump(self.PR,output_file)
            pickle.dump(self.IRL,output_file)
            pickle.dump(list(self.reward_list),output_file)
            pickle.dump(list(self.loss_list),output_file)
            pickle.dump(self.total_episode.value,output_file)
            output_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save(path)
                        self.avg_reward=avg_reward
        return
    
    
    def save(self,path):
        output_file=open(path,'wb')
        self.update_nn_param()
        self.nn.opt_counter=None
        self.nn.ec=self.nn.ec[0]
        self.nn.bc=self.nn.bc[0]
        self._episode_counter=list(self._episode_counter)
        self._batch_counter=list(self._batch_counter)
        self.nn.optimizer.convert_to_list()
        pickle.dump(self.nn,output_file)
        if self.save_data:
            pickle.dump(self.state_pool,output_file)
            pickle.dump(self.action_pool,output_file)
            pickle.dump(self.next_state_pool,output_file)
            pickle.dump(self.reward_pool,output_file)
            pickle.dump(self.done_pool,output_file)
        pickle.dump(self.policy,output_file)
        pickle.dump(self.noise,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(np.array(self.step_counter,dtype='int32'),output_file)
        pickle.dump(self.update_steps,output_file)
        pickle.dump(self.trial_count,output_file)
        pickle.dump(self.criterion,output_file)
        pickle.dump(self.PPO,output_file)
        pickle.dump(self.HER,output_file)
        pickle.dump(self.MARL,output_file)
        pickle.dump(self.PR,output_file)
        pickle.dump(self.IRL,output_file)
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
        if self.save_data:
            self.state_pool=pickle.load(input_file)
            self.action_pool=pickle.load(input_file)
            self.next_state_pool=pickle.load(input_file)
            self.reward_pool=pickle.load(input_file)
            self.done_pool=pickle.load(input_file)
        self.policy=pickle.load(input_file)
        self.noise=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.step_counter=pickle.load(input_file)
        self.step_counter=Array('i',self.step_counter)
        self.update_steps=pickle.load(input_file)
        self.trial_count=pickle.load(input_file)
        self.criterion=pickle.load(input_file)
        self.PPO=pickle.load(input_file)
        self.HER=pickle.load(input_file)
        self.MARL=pickle.load(input_file)
        self.PR=pickle.load(input_file)
        self.IRL=pickle.load(input_file)
        self.reward_list[:]=pickle.load(input_file)
        self.loss_list[:]=pickle.load(input_file)
        self.total_episode.value=pickle.load(input_file)
        self.episode_.value=self.total_episode.value
        input_file.close()
        return
