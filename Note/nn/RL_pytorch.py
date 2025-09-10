import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from Note.RL import rl
from Note.RL.rl.prioritized_replay import pr
from multiprocessing import Array
import numpy as np
import numpy.ctypeslib as npc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import statistics
import pickle
import os
import time


class RL_pytorch:
    def __init__(self):
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.state_pool_list=[]
        self.reward_list=[]
        self.step_counter=0
        self.store_counter=0
        self.prioritized_replay=pr()
        self.seed=7
        self.path=None
        self.save_freq=1
        self.save_freq_=None
        self.max_save_files=None
        self.save_best_only=False
        self.save_param_only=False
        self.path_list=[]
        self.loss=None
        self.loss_list=[]
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def set(self,policy=None,noise=None,pool_size=None,batch=None,num_updates=None,num_steps=None,update_batches=None,update_steps=None,trial_count=None,criterion=None,PPO=False,HER=False,MARL=False,PR=False,IRL=False,epsilon=None,initial_ratio=1.0,initial_TD=7,lambda_=0.5,alpha=0.7):
        self.policy=policy
        self.noise=noise
        self.pool_size=pool_size
        self.batch=batch
        self.num_updates=num_updates
        self.num_steps=num_steps
        self.update_batches=update_batches
        self.update_steps=update_steps
        self.trial_count=trial_count
        self.criterion=criterion
        self.PPO=PPO
        self.HER=HER
        self.MARL=MARL
        self.PR=PR
        self.IRL=IRL
        self.epsilon=epsilon
        if PPO:
            self.prioritized_replay.PPO=PPO
            self.initial_ratio=initial_ratio
            self.prioritized_replay.ratio=initial_ratio
            self.initial_TD=initial_TD
            self.prioritized_replay.TD=initial_TD
        else:
            self.initial_TD=initial_TD
            self.prioritized_replay.TD=initial_TD
        self.lambda_=lambda_
        self.alpha=alpha
        return
    
    
    def pool(self,s,a,next_s,r,done,index=None):
        if self.pool_network==True:
            if self.state_pool_list[index] is None:
                self.state_pool_list[index]=s
                self.action_pool_list[index]=np.expand_dims(a,axis=0)
                self.next_state_pool_list[index]=np.expand_dims(next_s,axis=0)
                self.reward_pool_list[index]=np.expand_dims(r,axis=0)
                self.done_pool_list[index]=np.expand_dims(done,axis=0)
            else:
                self.state_pool_list[index]=np.concatenate((self.state_pool_list[index],s),0)
                self.action_pool_list[index]=np.concatenate((self.action_pool_list[index],np.expand_dims(a,axis=0)),0)
                self.next_state_pool_list[index]=np.concatenate((self.next_state_pool_list[index],np.expand_dims(next_s,axis=0)),0)
                self.reward_pool_list[index]=np.concatenate((self.reward_pool_list[index],np.expand_dims(r,axis=0)),0)
                self.done_pool_list[index]=np.concatenate((self.done_pool[7],np.expand_dims(done,axis=0)),0)
            if self.clearing_freq!=None:
                self.store_counter[index]+=1
                if self.store_counter[index]%self.clearing_freq==0:
                    self.state_pool_list[index]=self.state_pool_list[index][self.window_size_:]
                    self.action_pool_list[index]=self.action_pool_list[index][self.window_size_:]
                    self.next_state_pool_list[index]=self.next_state_pool_list[index][self.window_size_:]
                    self.reward_pool_list[index]=self.reward_pool_list[index][self.window_size_:]
                    self.done_pool_list[index]=self.done_pool_list[index][self.window_size_:]
                    if self.PR:
                        if self.PPO:
                            self.ratio_list[index]=self.ratio_list[index][self.window_size_:]
                            self.TD_list[index]=self.TD_list[index][self.window_size_:]
                        else:
                            self.TD_list[index]=self.TD_list[index][self.window_size_:]
            if len(self.state_pool_list[index])>math.ceil(self.pool_size/self.processes):
                if type(self.window_size)!=int:
                    window_size=int(self.window_size(index))
                else:
                    window_size=self.window_size
                if window_size!=None:
                    self.state_pool_list[index]=self.state_pool_list[index][window_size:]
                    self.action_pool_list[index]=self.action_pool_list[index][window_size:]
                    self.next_state_pool_list[index]=self.next_state_pool_list[index][window_size:]
                    self.reward_pool_list[index]=self.reward_pool_list[index][window_size:]
                    self.done_pool_list[index]=self.done_pool_list[index][window_size:]
                    if self.PR:
                        if self.PPO:
                            self.ratio_list[index]=self.ratio_list[index][window_size:]
                            self.TD_list[index]=self.TD_list[index][window_size:]
                        else:
                            self.TD_list[index]=self.TD_list[index][window_size:]
                else:
                    self.state_pool_list[index]=self.state_pool_list[index][1:]
                    self.action_pool_list[index]=self.action_pool_list[index][1:]
                    self.next_state_pool_list[index]=self.next_state_pool_list[index][1:]
                    self.reward_pool_list[index]=self.reward_pool_list[index][1:]
                    self.done_pool_list[index]=self.done_pool_list[index][1:]
                    if self.PR:
                        if self.PPO:
                            self.ratio_list[index]=self.ratio_list[index][1:]
                            self.TD_list[index]=self.TD_list[index][1:]
                        else:
                            self.TD_list[index]=self.TD_list[index][1:]
        else:
            if self.state_pool is None:
                self.state_pool=s
                self.action_pool=np.expand_dims(a,axis=0)
                self.next_state_pool=np.expand_dims(next_s,axis=0)
                self.reward_pool=np.expand_dims(r,axis=0)
                self.done_pool=np.expand_dims(done,axis=0)
            else:
                self.state_pool=np.concatenate((self.state_pool,s),0)
                self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0)
                self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(next_s,axis=0)),0)
                self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0)
                self.done_pool=np.concatenate((self.done_pool,np.expand_dims(done,axis=0)),0)
            if self.clearing_freq!=None:
                self.store_counter+=1
                if self.store_counter%self.clearing_freq==0:
                    self.state_pool=self.state_pool[self.window_size_:]
                    self.action_pool=self.action_pool[self.window_size_:]
                    self.next_state_pool=self.next_state_pool[self.window_size_:]
                    self.reward_pool=self.reward_pool[self.window_size_:]
                    self.done_pool=self.done_pool[self.window_size_:]
                    if self.PR:
                        if self.PPO:
                            self.prioritized_replay.ratio=self.prioritized_replay.ratio[self.window_size_:]
                            self.prioritized_replay.TD=self.prioritized_replay.TD[self.window_size_:]
                        else:
                            self.prioritized_replay.TD=self.prioritized_replay.TD[self.window_size_:]
            if len(self.state_pool)>self.pool_size:
                if type(self.window_size)!=int:
                    window_size=self.window_size()
                else:
                    window_size=self.window_size
                if window_size!=None:
                    self.state_pool=self.state_pool[window_size:]
                    self.action_pool=self.action_pool[window_size:]
                    self.next_state_pool=self.next_state_pool[window_size:]
                    self.reward_pool=self.reward_pool[window_size:]
                    self.done_pool=self.done_pool[window_size:]
                    if self.PR:
                        if self.PPO:
                            self.prioritized_replay.ratio=self.prioritized_replay.ratio[window_size:]
                            self.prioritized_replay.TD=self.prioritized_replay.TD[window_size:]
                        else:
                            self.prioritized_replay.TD=self.prioritized_replay.TD[window_size:]
                else:
                    self.state_pool=self.state_pool[1:]
                    self.action_pool=self.action_pool[1:]
                    self.next_state_pool=self.next_state_pool[1:]
                    self.reward_pool=self.reward_pool[1:]
                    self.done_pool=self.done_pool[1:]
                    if self.PR:
                        if self.PPO:
                            self.prioritized_replay.ratio=self.prioritized_replay.ratio[1:]
                            self.prioritized_replay.TD=self.prioritized_replay.TD[1:]
                        else:
                            self.prioritized_replay.TD=self.prioritized_replay.TD[1:]
        return
    
    
    def select_action(self,s,i=None):
        if self.jit_compile==True:
            output=self.forward(s,i)
        else:
            output=self.forward_(s,i)
        if self.policy!=None:
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
                a=self.policy.select_action(output, self.step_counter)
            elif isinstance(self.policy, rl.GreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.MaxBoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannGumbelQPolicy):
                a=self.policy.select_action(output, self.step_counter)
        elif self.noise!=None:
            if self.IRL!=True:
                a=(output+self.noise.sample()).numpy()
            else:
                a=(output[1]+self.noise.sample()).numpy()
        if self.IRL!=True:
            return a
        else:
            return [output[0],a]
    
    
    def env_(self,a=None,initial=None,p=None):
        if initial==True:
            if self.pool_network==True:
                state=self.env[p].reset(seed=self.seed)
                return state
            else:
                state=self.env.reset(seed=self.seed)
                return state 
        else:
            if self.pool_network==True:
                next_state,reward,done,_=self.env[p].step(a)
                return next_state,reward,done
            else:
                next_state,reward,done,_=self.env.step(a)
                return next_state,reward,done
    
    
    def compute_ess_from_weights(self, weights):
        p = weights / (weights.sum())
        ess = 1.0 / (np.sum(p * p))
        return float(ess)


    def adjust_window_size(self, p=None, scale=1.0, smooth_alpha=0.2):
        if self.pool_network==True:
            if not hasattr(self, 'ema_ess'):
                self.ema_ess = [None] * self.processes
            
            if self.PPO:
                scores = self.lambda_ * self.TD_list[p] + (1.0-self.lambda_) * np.abs(self.ratio_list[p] - 1.0)
                weights = np.pow(scores + 1e-7, self.alpha)
            else:
                weights = np.pow(self.TD_list[p] + 1e-7, self.alpha)
    
            ess = self.compute_ess_from_weights(weights)
    
            if self.ema_ess[p] is None:
                ema = ess
            else:
                ema = smooth_alpha * ess + (1.0 - smooth_alpha) * self.ema_ess[p]
            self.ema_ess[p] = ema
        else:
            if not hasattr(self, 'ema_ess'):
                self.ema_ess = None
            
            if self.PPO:
                scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * np.abs(self.prioritized_replay.ratio - 1.0)
                weights = np.pow(scores + 1e-7, self.alpha)
            else:
                weights = np.pow(self.prioritized_replay.TD + 1e-7, self.alpha)
            
            ess = self.compute_ess_from_weights(weights)
            
            if self.ema_ess is None:
                ema = ess
            else:
                ema = smooth_alpha * ess + (1.0 - smooth_alpha) * self.ema_ess
            self.ema_ess = ema
            
        desired_keep = np.clip(int(ema * scale), 1, len(weights) - 1)
        
        window_size = len(weights) - desired_keep
        return window_size
    
    
    def adjust_batch_size(self, scale=1.0, smooth_alpha=0.2, min_batch=None, max_batch=None, align=None):
        if not hasattr(self, 'ema_ess'):
            self.ema_ess = None
        
        if self.PPO:
            scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * np.abs(self.prioritized_replay.ratio - 1.0)
            weights = np.pow(scores + 1e-7, self.alpha)
        else:
            weights = np.pow(self.prioritized_replay.TD + 1e-7, self.alpha)
        
        ess = self.compute_ess_from_weights(weights)
        
        if self.ema_ess is None:
            ema = ess
        else:
            ema = smooth_alpha * ess + (1.0 - smooth_alpha) * self.ema_ess
        self.ema_ess = ema
            
        buf_len = len(weights)
        if min_batch is None:
            cur_batch = self.batch
            min_batch = max(1, cur_batch // 2)
        if max_batch is None:
            max_batch = max(1, buf_len)
        
        batch = int(max(1, round(ema * float(scale))))
        batch = int(np.clip(batch, min_batch, max_batch))
        
        if align is None:
            align = self.batch
        new_batch = batch - (batch % align)
        new_batch = max(1, new_batch)
    
        new_batch = int(min(new_batch, buf_len))
        return int(new_batch)
    
    
    def data_func(self):
        if self.PR:
            if self.processes_pr!=None:
                process_list=[]
                for p in range(self.processes_pr):
                    process=mp.Process(target=self.get_batch_in_parallel,args=(p,))
                    process.start()
                    process_list.append(process)
                for process in process_list:
                    process.join()
                s = np.array(self.state_list)
                a = np.array(self.action_list)
                next_s = np.array(self.next_state_list)
                r = np.array(self.reward_list)
                d = np.array(self.done_list)
            else:
                s,a,next_s,r,d=self.prioritized_replay.sample(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.epsilon,self.lambda_,self.alpha,self.batch)
        elif self.HER:
            if self.processes_her!=None:
                process_list=[]
                for p in range(self.processes_her):
                    process=mp.Process(target=self.get_batch_in_parallel,args=(p,))
                    process.start()
                    process_list.append(process)
                for process in process_list:
                    process.join()
                s = np.array(self.state_list)
                a = np.array(self.action_list)
                next_s = np.array(self.next_state_list)
                r = np.array(self.reward_list)
                d = np.array(self.done_list)
            else:
                s = []
                a = []
                next_s = []
                r = []
                d = []
                for _ in range(self.batch):
                    step_state = np.random.randint(0, len(self.state_pool)-1)
                    step_goal = np.random.randint(step_state+1, step_state+np.argmax(self.done_pool[step_state+1:])+2)
                    state = self.state_pool[step_state]
                    next_state = self.next_state_pool[step_state]
                    action = self.action_pool[step_state]
                    goal = self.state_pool[step_goal]
                    reward, done = self.reward_done_func(next_state, goal)
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
    
    
    def train_step(self, train_data, optimizer):
        loss = self.__call__(*train_data)
        if type(optimizer)!=list:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            for i in range(len(optimizer)):
                optimizer[i].zero_grad()
            loss.backward()
            for i in range(len(optimizer)):
                optimizer[i].step()
        return loss
    
    
    def train1(self, optimizer):
        loss=0
        self.step_counter+=1
        batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
        if len(self.state_pool)%self.batch!=0:
            batches+=1
        if self.PR==True or self.HER==True:
            for j in range(batches):
                if self.batch_counter%self.num_updates==0:
                    break
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer)
                self.prioritized_replay.update()
                self.batch_counter+=1
                if hasattr(self, 'batch_size_fn') and len(self.state_pool)>=self.pool_size_:
                    self.batch = self.batch_size_fn()
                if self.pool_network==True:
                    if self.batch_counter%self.update_batches==0:
                        self.update_param()
                        if not hasattr(self,'window_size_fn'):
                            if self.PPO:
                                window_size=self.window_size_ppo
                            else:
                                window_size=self.window_size_pr
                        for p in range(self.processes):
                            if hasattr(self,'window_size_fn'):
                                window_size=int(self.window_size_fn(p))
                            if window_size!=None and len(self.state_pool_list[p])>window_size:
                                self.state_pool_list[p]=self.state_pool_list[p][window_size:]
                                self.action_pool_list[p]=self.action_pool_list[p][window_size:]
                                self.next_state_pool_list[p]=self.action_pool_list[p][window_size:]
                                self.reward_pool_list[p]=self.action_pool_list[p][window_size:]
                                self.done_pool_list[p]=self.action_pool_list[p][window_size:]
                                if self.PPO:
                                    self.ratio_list[p]=self.ratio_list[p][window_size:]
                                self.TD_list[p]=self.TD_list[p][window_size:]
                        if self.processes_her==None and self.processes_pr==None:
                            self.state_pool=np.concatenate(self.state_pool_list)
                            self.action_pool=np.concatenate(self.action_pool_list)
                            self.next_state_pool=np.concatenate(self.next_state_pool_list)
                            self.reward_pool=np.concatenate(self.reward_pool_list)
                            self.done_pool=np.concatenate(self.done_pool_list)
                        else:
                            self.state_pool[7]=np.concatenate(self.state_pool_list)
                            self.action_pool[7]=np.concatenate(self.action_pool_list)
                            self.next_state_pool[7]=np.concatenate(self.next_state_pool_list)
                            self.reward_pool[7]=np.concatenate(self.reward_pool_list)
                            self.done_pool[7]=np.concatenate(self.done_pool_list)
                        if self.PPO:
                            self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                            self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                        else:
                            self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                        if hasattr(self, 'batch_size_fn') and len(self.state_pool)>=self.pool_size_:
                            self.batch = self.batch_size_fn()
            if len(self.state_pool)%self.batch!=0:
                if self.batch_counter%self.num_updates==0:
                    return loss.detach().numpy()/batches
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer)
                self.prioritized_replay.update()
                self.batch_counter+=1
                if hasattr(self, 'batch_size_fn') and len(self.state_pool)>=self.pool_size_:
                    self.batch = self.batch_size_fn()
                if self.pool_network==True:
                    if self.batch_counter%self.update_batches==0:
                        self.update_param()
                        if not hasattr(self,'window_size_fn'):
                            if self.PPO:
                                window_size=self.window_size_ppo
                            else:
                                window_size=self.window_size_pr
                        for p in range(self.processes):
                            if hasattr(self,'window_size_fn'):
                                window_size=int(self.window_size_fn(p))
                            if window_size!=None and len(self.state_pool_list[p])>window_size:
                                self.state_pool_list[p]=self.state_pool_list[p][window_size:]
                                self.action_pool_list[p]=self.action_pool_list[p][window_size:]
                                self.next_state_pool_list[p]=self.action_pool_list[p][window_size:]
                                self.reward_pool_list[p]=self.action_pool_list[p][window_size:]
                                self.done_pool_list[p]=self.action_pool_list[p][window_size:]
                                if self.PPO:
                                    self.ratio_list[p]=self.ratio_list[p][window_size:]
                                self.TD_list[p]=self.TD_list[p][window_size:]
                        if self.processes_her==None and self.processes_pr==None:
                            self.state_pool=np.concatenate(self.state_pool_list)
                            self.action_pool=np.concatenate(self.action_pool_list)
                            self.next_state_pool=np.concatenate(self.next_state_pool_list)
                            self.reward_pool=np.concatenate(self.reward_pool_list)
                            self.done_pool=np.concatenate(self.done_pool_list)
                        else:
                            self.state_pool[7]=np.concatenate(self.state_pool_list)
                            self.action_pool[7]=np.concatenate(self.action_pool_list)
                            self.next_state_pool[7]=np.concatenate(self.next_state_pool_list)
                            self.reward_pool[7]=np.concatenate(self.reward_pool_list)
                            self.done_pool[7]=np.concatenate(self.done_pool_list)
                        if self.PPO:
                            self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                            self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                        else:
                            self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                        if hasattr(self, 'batch_size_fn') and len(self.state_pool)>=self.pool_size_:
                            self.batch = self.batch_size_fn()
        else:
            if self.pool_network==True:
                train_ds=DataLoader((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool),batch_size=self.batch)
            else:
                train_ds=DataLoader((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool),batch_size=self.batch,shuffle=True)
            for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer)
                self.batch_counter+=1
                if self.pool_network==True:
                    if self.batch_counter%self.update_batches==0:
                        self.update_param()
                        if self.PPO:
                            for p in range(self.processes):
                                self.state_pool_list[p]=None
                                self.action_pool_list[p]=None
                                self.next_state_pool_list[p]=None
                                self.reward_pool_list[p]=None
                                self.done_pool_list[p]=None
        if self.update_steps!=None:
            if hasattr(self, 'batch_size_fn') and len(self.state_pool)>=self.pool_size_:
                self.batch = self.batch_size_fn()
            if self.step_counter%self.update_steps==0:
                self.update_param()
                if self.PR:
                    if not hasattr(self,'window_size_fn'):
                        if self.PPO:
                            window_size=self.window_size_ppo
                        else:
                            window_size=self.window_size_pr
                    if hasattr(self,'window_size_fn'):
                        window_size=int(self.window_size_fn())
                    if window_size!=None and len(self.state_pool)>window_size:
                        self.state_pool=self.state_pool[window_size:]
                        self.action_pool=self.action_pool[window_size:]
                        self.next_state_pool=self.action_pool[window_size:]
                        self.reward_pool=self.action_pool[window_size:]
                        self.done_pool=self.action_pool[window_size:]
                        if self.PPO:
                            self.prioritized_replay.ratio=self.prioritized_replay.ratio[window_size:]
                        self.prioritized_replay.TD=self.prioritized_replay.TD[window_size:]
                    if hasattr(self, 'batch_size_fn') and len(self.state_pool)>=self.pool_size_:
                        self.batch = self.batch_size_fn()
                else:
                    self.state_pool=None
                    self.action_pool=None
                    self.next_state_pool=None
                    self.reward_pool=None
                    self.done_pool=None
        else:
            self.update_param()
        return loss.detach().numpy()/batches
    
    
    def train2(self, optimizer):
        self.reward=0
        s=self.env_(initial=True)
        reward=0
        counter=0
        while True:
            s=np.expand_dims(s,axis=0)
            if self.MARL!=True:
                a=self.select_action(s)
            else:
                a=[]
                for i in len(s[0]):
                    s=np.expand_dims(s[0][i],axis=0)
                    a.append([self.select_action(s,i)])
                a=np.array(a)
            next_s,r,done=self.env_(a)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            if self.PR==True:
                if self.PPO:
                    if len(self.state_pool)>1:
                        self.prioritized_replay.ratio=np.append(self.prioritized_replay.ratio,np.max(self.prioritized_replay.ratio))
                    if len(self.state_pool)>1:
                        self.prioritized_replay.TD=np.append(self.prioritized_replay.ratio,np.max(self.prioritized_replay.TD))
                else:
                    if len(self.state_pool)>1:
                        self.prioritized_replay.TD=np.append(self.prioritized_replay.ratio,np.max(self.prioritized_replay.TD))
            if self.MARL==True:
                r,done=self.reward_done_func_ma(r,done)
            self.reward=r+self.reward
            if self.num_steps!=None:
                if counter==0:
                    next_s_=next_s
                    done_=done
                counter+=1
                reward=r+reward
                if counter%self.num_steps==0 or done:
                    self.pool(s,a,next_s,reward,done)
                    reward=0
            else:
                self.pool(s,a,next_s,r,done)
            if not self.PR and self.num_updates!=None:
                state_pool=self.state_pool
                action_pool=self.action_pool
                next_state_pool=self.next_state_pool
                reward_pool=self.reward_pool
                done_pool=self.done_pool
                if len(self.state_pool)>=self.pool_size_:
                    idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                else:
                    idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                self.state_pool=self.state_pool[idx]
                self.action_pool=self.action_pool[idx]
                self.next_state_pool=self.action_pool[idx]
                self.reward_pool=self.action_pool[idx]
                self.done_pool=self.action_pool[idx]
            loss=self.train1(optimizer)
            if not self.PR and self.num_updates!=None:
                self.state_pool=state_pool
                self.action_pool=action_pool
                self.next_state_pool=next_state_pool
                self.reward_pool=reward_pool
                self.done_pool=done_pool
            if (self.num_steps==None and done) or (self.num_steps!=None and done_):
                self.reward_list.append(self.reward)
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
                if len(self.state_pool)<self.batch:
                    s=self.env_(initial=True)
                    continue
                return loss
            s=next_s
            if (self.num_steps!=None and counter%self.num_steps==0) or (self.num_steps!=None and done):
                s=next_s_
    
    
    def get_batch_in_parallel(self,p):
        s = []
        a = []
        next_s = []
        r = []
        d = []
        if self.HER==True:
            for _ in range(int(self.batch/self.processes_her)):
                step_state = np.random.randint(0, len(self.state_pool[7])-1)
                step_goal = np.random.randint(step_state+1, step_state+np.argmax(self.done_pool[7][step_state+1:])+2)
                state = self.state_pool[7][step_state]
                next_state = self.next_state_pool[7][step_state]
                action = self.action_pool[7][step_state]
                goal = self.state_pool[7][step_goal]
                reward, done = self.reward_done_func(next_state, goal)
                state = np.hstack((state, goal))
                next_state = np.hstack((next_state, goal))
                s.append(state)
                a.append(action)
                next_s.append(next_state)
                r.append(reward)
                d.append(done)
        elif self.PR==True:
            for _ in range(int(self.batch/self.processes_pr)):
                state,action,next_state,reward,done=self.prioritized_replay.sample(self.state_pool[7],self.action_pool[7],self.next_state_pool[7],self.reward_pool[7],self.done_pool[7],self.epsilon,self.lambda_,self.alpha,int(self.batch/self.processes_pr))
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
        self.state_list[p]=s
        self.action_list[p]=a
        self.next_state_list[p]=next_s
        self.reward_list[p]=r
        self.done_list[p]=d
        return
    
    
    def modify_ratio_TD(self):
        if self.PR==True:
            for p in range(self.processes):
                if self.prioritized_replay.ratio is not None:
                    if p==0:
                        self.ratio_list[p]=self.prioritized_replay.ratio[0:len(self.ratio_list[p])]
                    else:
                        index1=0
                        index2=0
                        for i in range(p):
                            index1+=len(self.ratio_list[i])
                        index2=index1+len(self.ratio_list[p])
                        self.ratio_list[p]=self.prioritized_replay.ratio[index1-1:index2]
                if self.prioritized_replay.TD is not None:
                    if p==0:
                        self.TD_list[p]=self.prioritized_replay.TD[0:len(self.TD_list[p])]
                    else:
                        index1=0
                        index2=0
                        for i in range(p):
                            index1+=len(self.TD_list[i])
                        index2=index1+len(self.TD_list[p])
                        self.TD_list[p]=self.prioritized_replay.TD[index1-1:index2]
        return
    
    
    def modify_TD(self):
        if self.PR==True:
            for p in range(self.processes):
                if self.prioritized_replay.TD is not None:
                    if p==0:
                        self.TD_list[p]=self.prioritized_replay.TD[0:len(self.TD_list[p])]
                    else:
                        index1=0
                        index2=0
                        for i in range(p):
                            index1+=len(self.TD_list[i])
                        index2=index1+len(self.TD_list[p])
                        self.TD_list[p]=self.prioritized_replay.TD[index1-1:index2]
        return
            
            
    def store_in_parallel(self,p,lock_list):
        self.reward[p]=0
        s=self.env_(initial=True,p=p)
        s=np.array(s)
        reward=0
        counter=0
        while True:
            if self.random or (self.PR!=True and self.HER!=True):
                if self.state_pool_list[p] is None:
                    index=p
                    self.inverse_len[index]=1
                else:
                    inverse_len=torch.tensor(self.inverse_len)
                    total_inverse=torch.sum(inverse_len)
                    prob=inverse_len/total_inverse
                    index=np.random.choice(self.processes,p=prob.numpy(),replace=False)
                    self.inverse_len[index]=1/(len(self.state_pool_list[index])+1)
            else:
                index=p
            s=np.expand_dims(s,axis=0)
            if self.MARL!=True:
                a=self.select_action(s)
            else:
                a=[]
                for i in len(s[0]):
                    s=np.expand_dims(s[0][i],axis=0)
                    a.append([self.select_action(s,i)])
                a=np.array(a)
            next_s,r,done=self.env_(a,p=p)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            if self.random or (self.PR!=True and self.HER!=True):
                lock_list[index].acquire()
                if self.num_steps!=None:
                    if counter==0:
                        next_s_=next_s
                        done_=done
                    counter+=1
                    reward=r+reward
                    if counter%self.num_steps==0 or done:
                        self.pool(s,a,next_s,reward,done,index)
                        reward=0
                else:
                    self.pool(s,a,next_s,r,done,index)
                lock_list[index].release()
            else:
                if self.num_steps!=None:
                    if counter==0:
                        next_s_=next_s
                        done_=done
                    counter+=1
                    reward=r+reward
                    if counter%self.num_steps==0 or done:
                        self.pool(s,a,next_s,reward,done,index)
                        reward=0
                else:
                    self.pool(s,a,next_s,r,done,index)
                if self.PR==True:
                    if self.PPO:
                        if len(self.state_pool_list[index])>1:
                            self.ratio_list[index]=np.append(self.ratio_list[index],np.max(self.prioritized_replay.ratio))
                        if len(self.state_pool_list[index])>1:
                            self.TD_list[index]=np.append(self.TD_list[index],np.max(self.prioritized_replay.TD))
                    else:
                        if len(self.state_pool_list[index])>1:
                            self.TD_list[index]=np.append(self.TD_list[index],np.max(self.prioritized_replay.TD))
            if self.MARL==True:
                r,done=self.reward_done_func_ma(r,done)
            self.reward[p]=r+self.reward[p]
            if (self.num_steps==None and done) or (self.num_steps!=None and done_):
                return
            s=next_s
            if (self.num_steps!=None and counter%self.num_steps==0) or (self.num_steps!=None and done):
                s=next_s_
    
    
    def prepare(self, lock_list):
        process_list=[]
        if self.PPO:
            self.modify_ratio_TD()
        else:
            self.modify_TD()
        while True:
            for p in range(self.processes):
                process=mp.Process(target=self.store_in_parallel,args=(p,lock_list))
                process.start()
                process_list.append(process)
            for process in process_list:
                process.join()
            if self.processes_her==None and self.processes_pr==None:
                self.state_pool=np.concatenate(self.state_pool_list)
                self.action_pool=np.concatenate(self.action_pool_list)
                self.next_state_pool=np.concatenate(self.next_state_pool_list)
                self.reward_pool=np.concatenate(self.reward_pool_list)
                self.done_pool=np.concatenate(self.done_pool_list)
                if len(self.state_pool)<self.batch:
                    continue
                if not self.PR and self.num_updates!=None:
                    if len(self.state_pool)>=self.pool_size_:
                        idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                    else:
                        idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                    self.state_pool=self.state_pool[idx]
                    self.action_pool=self.action_pool[idx]
                    self.next_state_pool=self.next_state_pool[idx]
                    self.reward_pool=self.reward_pool[idx]
                    self.done_pool=self.done_pool[idx]
            else:
                self.state_pool[7]=np.concatenate(self.state_pool_list)
                self.action_pool[7]=np.concatenate(self.action_pool_list)
                self.next_state_pool[7]=np.concatenate(self.next_state_pool_list)
                self.reward_pool[7]=np.concatenate(self.reward_pool_list)
                self.done_pool[7]=np.concatenate(self.done_pool_list)
                if len(self.state_pool[7])<self.batch:
                    continue
                if not self.PR and self.num_updates!=None:
                    if len(self.state_pool)>=self.pool_size_:
                        idx=np.random.choice(self.state_pool[7].shape[0], size=self.pool_size_, replace=False)
                    else:
                        idx=np.random.choice(self.state_pool[7].shape[0], size=self.state_pool[7].shape[0], replace=False)
                    self.state_pool[7]=self.state_pool[7][idx]
                    self.action_pool[7]=self.action_pool[7][idx]
                    self.next_state_pool[7]=self.next_state_pool[7][idx]
                    self.reward_pool[7]=self.reward_pool[7][idx]
                    self.done_pool[7]=self.done_pool[7][idx]
            if self.processes_her==None and self.processes_pr==None:
                if len(self.state_pool)>=self.batch:
                    break
            else:
                if len(self.state_pool[7])>=self.batch:
                    break
        if self.PR==True:
            if self.PPO:
                self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
            else:
                self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
            if hasattr(self, 'batch_size_fn') and len(self.state_pool)>=self.pool_size_:
                self.batch = self.batch_size_fn()
        self.reward_list.append(np.mean(npc.as_array(self.reward.get_obj())))
        if len(self.reward_list)>self.trial_count:
            del self.reward_list[0]
    
    
    def train(self, optimizer, episodes=None, pool_network=True, processes=None, processes_her=None, processes_pr=None, window_size=None, clearing_freq=None, window_size_=None, window_size_ppo=None, window_size_pr=None, random=False, save_data=True, p=None):
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if episodes%10!=0:
            p=episodes-episodes%self.p
            p=int(p/self.p)
        else:
            p=episodes/(self.p+1)
            p=int(p)
        if p==0:
            p=1
        self.pool_network=pool_network
        self.processes=processes
        self.processes_her=processes_her
        self.processes_pr=processes_pr
        self.window_size=window_size
        self.clearing_freq=clearing_freq
        self.window_size_=window_size_
        self.window_size_ppo=window_size_ppo
        self.window_size_pr=window_size_pr
        self.random=random
        if self.num_updates!=None:
            self.pool_size_=self.num_updates*self.batch
        self.save_data=save_data
        if pool_network==True:
            manager=mp.Manager()
            self.env=manager.list(self.env)
            if save_data and len(self.state_pool_list)!=0 and self.state_pool_list[0] is not None:
                self.state_pool_list=manager.list(self.state_pool_list)
                self.action_pool_list=manager.list(self.action_pool_list)
                self.next_state_pool_list=manager.list(self.next_state_pool_list)
                self.reward_pool_list=manager.list(self.reward_pool_list)
                self.done_pool_list=manager.list(self.done_pool_list)
                if self.clearing_freq!=None:
                    self.store_counter=manager.list(self.store_counter)
            else:
                self.state_pool_list=manager.list()
                self.action_pool_list=manager.list()
                self.next_state_pool_list=manager.list()
                self.reward_pool_list=manager.list()
                self.done_pool_list=manager.list()
                self.inverse_len=manager.list([0 for _ in range(processes)])
                if self.clearing_freq!=None:
                    self.store_counter=manager.list()
            if not save_data or len(self.state_pool_list)==0:
                for _ in range(processes):
                    self.state_pool_list.append(None)
                    self.action_pool_list.append(None)
                    self.next_state_pool_list.append(None)
                    self.reward_pool_list.append(None)
                    self.done_pool_list.append(None)
                    if self.clearing_freq!=None:
                        self.store_counter.append(0)
            self.reward=np.zeros(processes,dtype='float32')
            self.reward=Array('f',self.reward)
            if self.HER!=True:
                lock_list=[mp.Lock() for _ in range(processes)]
            else:
                lock_list=None
            if self.PR==True:
                if self.PPO:
                    self.ratio_list=manager.list()
                    self.TD_list=manager.list()
                    for _ in range(processes):
                        self.ratio_list.append(np.array(self.initial_ratio).astype('float32'))
                        self.TD_list.append(np.array(self.initial_TD).astype('float32'))
                    self.prioritized_replay.ratio=None
                    self.prioritized_replay.TD=None
                else:
                    self.TD_list=manager.list()
                    for _ in range(processes):
                        self.TD_list.append(np.array(self.initial_TD).astype('float32'))
                    self.prioritized_replay.TD=None
            if processes_her!=None or processes_pr!=None:
                self.state_pool=manager.dict()
                self.action_pool=manager.dict()
                self.next_state_pool=manager.dict()
                self.reward_pool=manager.dict()
                self.done_pool=manager.dict()
                self.state_list=manager.list()
                self.action_list=manager.list()
                self.next_state_list=manager.list()
                self.reward_list=manager.list()
                self.done_list=manager.list()
                if processes_her!=None:
                    for _ in range(processes_her):
                        self.state_list.append(None)
                        self.action_list.append(None)
                        self.next_state_list.append(None)
                        self.reward_list.append(None)
                        self.done_list.append(None)
                else:
                    for _ in range(processes_pr):
                        self.state_list.append(None)
                        self.action_list.append(None)
                        self.next_state_list.append(None)
                        self.reward_list.append(None)
                        self.done_list.append(None)
        self.optimizer=optimizer
        if episodes!=None:
            for i in range(episodes):
                t1=time.time()
                if pool_network==True:
                    self.prepare(lock_list)
                    loss=self.train1(self.optimizer)
                else:
                    loss=self.train2(self.optimizer)
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                if pool_network==True:
                    self.prepare(lock_list)
                    loss=self.train1(self.optimizer)
                else:
                    loss=self.train2(self.optimizer)
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        time_=self.time-int(self.time)
        if time_<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
        self.total_time+=self.time
        print('time:{0}s'.format(self.time))
        return
    
    
    def run_agent(self, max_steps, seed=None):
        state_history = []

        steps = 0
        reward_ = 0
        if seed==None:
            state = self.env.reset()
        else:
            state = self.env.reset(seed=seed)
        for step in range(max_steps):
            if self.noise==None:
                action = np.argmax(self.action(state))
            else:
                action = self.action(state).numpy()
            next_state, reward, done, _ = self.env.step(action)
            state_history.append(state)
            steps+=1
            reward_+=reward
            if done:
                break
            state = next_state
        
        return state_history,reward_,steps
    
    
    def animate_agent(self, max_steps, mode='rgb_array', save_path=None, fps=None, writer='imagemagick'):
        state_history,reward,steps = self.run_agent(max_steps)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        self.env.reset()
        img = ax.imshow(self.env.render(mode=mode))

        def update(frame):
            img.set_array(self.env.render(mode=mode))
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=state_history, blit=True)
        plt.show()
        
        print('steps:{0}'.format(steps))
        print('reward:{0}'.format(reward))
        
        if save_path!=None:
            ani.save(save_path, writer=writer, fps=fps)
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('reward:{0:.4f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('loss:{0:.4f}'.format(self.loss_list[-1]))
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
    
    
    def save_param_(self,path):
        if self.save_best_only==False:
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(path,'wb')
            else:
                if self.train_acc!=None and self.test_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch,self.train_acc,self.test_acc))
                elif self.train_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.train_acc))
                else:
                    path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_epoch))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self.param,output_file)
            output_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save_param(path)
                        self.avg_reward=avg_reward
        return
    
    
    def save_param(self,path):
        output_file=open(path,'wb')
        pickle.dump(self.param,output_file)
        output_file.close()
        return
    
    
    def restore_param(self,path):
        input_file=open(path,'rb')
        param=pickle.load(input_file)
        for target_param,source_param in zip(self.param,param):
            target_param.data=source_param.data
        input_file.close()
        return
    
    
    def save_(self,path):
        if self.pool_network and not self.save_data:
            state_pool_list=[]
            action_pool_list=[]
            next_state_pool_list=[]
            reward_pool_list=[]
            done_pool_list=[]
            if self.processes_her==None and self.processes_pr==None:
                self.state_pool=None
                self.action_pool=None
                self.next_state_pool=None
                self.reward_pool=None
                self.done_pool=None
            else:
                self.state_pool[7]=None
                self.action_pool[7]=None
                self.next_state_pool[7]=None
                self.reward_pool[7]=None
                self.done_pool[7]=None
            for i in range(self.processes):
                state_pool_list.append(self.state_pool_list[i])
                action_pool_list.append(self.action_pool_list[i])
                next_state_pool_list.append(self.next_state_pool_list[i])
                reward_pool_list.append(self.reward_pool_list[i])
                done_pool_list.append(self.done_pool_list[i])
                self.state_pool_list[i]=None
                self.action_pool_list[i]=None
                self.next_state_pool_list[i]=None
                self.reward_pool_list[i]=None
                self.done_pool_list[i]=None
        if self.save_best_only==False:
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(path,'wb')
            else:
                if self.train_acc!=None and self.test_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch,self.train_acc,self.test_acc))
                elif self.train_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.train_acc))
                else:
                    path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_epoch))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self,output_file)
            output_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save(path)
                        self.avg_reward=avg_reward
        if self.pool_network and not self.save_data:
            for i in range(self.processes):
                self.state_pool_list[i]=state_pool_list[i]
                self.action_pool_list[i]=action_pool_list[i]
                self.next_state_pool_list[i]=next_state_pool_list[i]
                self.reward_pool_list[i]=reward_pool_list[i]
                self.done_pool_list[i]=done_pool_list[i]
        return
    
    
    def save(self,path):
        if self.pool_network and not self.save_data:
            state_pool_list=[]
            action_pool_list=[]
            next_state_pool_list=[]
            reward_pool_list=[]
            done_pool_list=[]
            if self.processes_her==None and self.processes_pr==None:
                self.state_pool=None
                self.action_pool=None
                self.next_state_pool=None
                self.reward_pool=None
                self.done_pool=None
            else:
                self.state_pool[7]=None
                self.action_pool[7]=None
                self.next_state_pool[7]=None
                self.reward_pool[7]=None
                self.done_pool[7]=None
            for i in range(self.processes):
                state_pool_list.append(self.state_pool_list[i])
                action_pool_list.append(self.action_pool_list[i])
                next_state_pool_list.append(self.next_state_pool_list[i])
                reward_pool_list.append(self.reward_pool_list[i])
                done_pool_list.append(self.done_pool_list[i])
                self.state_pool_list[i]=None
                self.action_pool_list[i]=None
                self.next_state_pool_list[i]=None
                self.reward_pool_list[i]=None
                self.done_pool_list[i]=None
        output_file=open(path,'wb')
        pickle.dump(self,output_file)
        output_file.close()
        if self.pool_network and not self.save_data:
            for i in range(self.processes):
                self.state_pool_list[i]=state_pool_list[i]
                self.action_pool_list[i]=action_pool_list[i]
                self.next_state_pool_list[i]=next_state_pool_list[i]
                self.reward_pool_list[i]=reward_pool_list[i]
                self.done_pool_list[i]=done_pool_list[i]
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        model=pickle.load(input_file)
        self.__dict__.update(model.__dict__)
        input_file.close()
        return
