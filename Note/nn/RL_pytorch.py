import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from Note.RL import rl
from Note.RL.rl.prioritized_replay import PR
import numpy as np
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
        self.prioritized_replay=PR()
        self.buffer_safety_factor=1
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
    
    
    def set(self,policy=None,noise=None,pool_size=None,batch=None,num_updates=None,num_steps=None,update_batches=None,update_steps=None,trial_count=None,criterion=None,PPO=False,HER=False,TRL=False,MARL=False,PR=False,IRL=False,initial_ratio=1.0,initial_TD=7,lambda_=0.5,alpha=0.7):
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
        self.TRL=TRL
        self.MARL=MARL
        self.PR=PR
        self.IRL=IRL
        self.initial_ratio=np.array(initial_ratio).astype('float32')
        self.initial_TD=np.array(initial_TD).astype('float32')
        if PR:
            if hasattr(self.prioritized_replay, 'sum_tree'):
                self.prioritized_replay.build(pool_size, alpha)
            if PPO:
                self.prioritized_replay.PPO=PPO
                self.prioritized_replay.ratio=self.initial_ratio
                self.prioritized_replay.TD=self.initial_TD
            else:
                self.prioritized_replay.TD=self.initial_TD
        self.lambda_=lambda_
        self.alpha=alpha
        return
    
    
    def pool(self,s,a,next_s,r,done,p=None):
        if self.pool_network==True:
            pos = self.write_indices[p]
            curr_len = self.pool_lengths[p]
            self._get_buffer(p, 'state')[pos] = np.asarray(s).reshape(self.state_shape)
            self._get_buffer(p, 'action')[pos] = np.asarray(a).reshape(self.action_shape)
            self._get_buffer(p, 'next_state')[pos] = np.asarray(next_s).reshape(self.next_state_shape)
            self._get_buffer(p, 'reward')[pos] = float(r)
            self._get_buffer(p, 'done')[pos] = float(done)
            if self.PR==True:
                if self.PPO:
                    self._get_buffer(p, 'ratio')[pos] = np.max(self._get_buffer(p, 'ratio')[:curr_len]) if curr_len > 0 else self.initial_ratio
                    self._get_buffer(p, 'TD')[pos] = self.initial_TD if curr_len == 0 else np.max(self._get_buffer(p, 'TD')[:curr_len])
                else:
                    self._get_buffer(p, 'TD')[pos] = self.initial_TD if curr_len == 0 else np.max(self._get_buffer(p, 'TD')[:curr_len])
                if hasattr(self.prioritized_replay, 'sum_tree'):
                    if self.PPO:
                        new_td = self.initial_TD if curr_len == 0 else np.max(self._get_buffer(p, 'TD')[:curr_len])
                        new_ratio = self.initial_ratio if curr_len == 0 else np.max(self._get_buffer(p, 'ratio')[:curr_len])
                        score = self.lambda_ * new_td + (1.0 - self.lambda_) * np.abs(new_ratio - 1.0)
                        new_prio = (score + 1e-7) ** self.alpha
                    else:
                        new_td = self.initial_TD if curr_len == 0 else np.max(self._get_buffer(p, 'TD')[:curr_len])
                        new_prio = (new_td + 1e-7) ** self.alpha
                    self.prioritized_replay.sum_tree.add(new_prio)
            self.write_indices[p] = pos + 1
            self.pool_lengths[p] = min(curr_len + 1, self.max_exp_per_proc)
            pos = self.write_indices[p]
            curr_len = self.pool_lengths[p]
            if not self.parallel_store_and_training or (self.parallel_store_and_training and not self.PR):
                if self.clearing_freq!=None:
                    self.store_counter[p]+=1
                    if self.store_counter[p]%self.clearing_freq==0:
                        self._get_buffer(p, 'state')[:curr_len-self.window_size_]=self._get_buffer(p, 'state')[self.window_size_:]
                        self._get_buffer(p, 'action')[:curr_len-self.window_size_]=self._get_buffer(p, 'action')[self.window_size_:]
                        self._get_buffer(p, 'next_state')[:curr_len-self.window_size_]=self._get_buffer(p, 'next_state')[self.window_size_:]
                        self._get_buffer(p, 'reward')[:curr_len-self.window_size_]=self._get_buffer(p, 'reward')[self.window_size_:]
                        self._get_buffer(p, 'done')[:curr_len-self.window_size_]=self._get_buffer(p, 'done')[self.window_size_:]
                        self.write_indices[p] = curr_len-self.window_size_
                        self.pool_lengths[p] = curr_len-self.window_size_
                        if self.PR:
                            if self.PPO:
                                self._get_buffer(p, 'ratio')[:curr_len-self.window_size_]=self._get_buffer(p, 'ratio')[self.window_size_:]
                                self._get_buffer(p, 'TD')[:curr_len-self.window_size_]=self._get_buffer(p, 'TD')[self.window_size_:]
                            else:
                                self._get_buffer(p, 'TD')[:curr_len-self.window_size_]=self._get_buffer(p, 'TD')[self.window_size_:]
                            if hasattr(self.prioritized_replay, 'sum_tree'):
                                self.prioritized_replay.rebuild()
                if curr_len==math.ceil(self.pool_size/self.processes):
                    if type(self.window_size)!=int:
                        window_size=int(self.window_size(p))
                    else:
                        window_size=self.window_size
                    if window_size!=None:
                        self._get_buffer(p, 'state')[:curr_len-window_size]=self._get_buffer(p, 'state')[window_size:]
                        self._get_buffer(p, 'action')[:curr_len-window_size]=self._get_buffer(p, 'action')[window_size:]
                        self._get_buffer(p, 'next_state')[:curr_len-window_size]=self._get_buffer(p, 'next_state')[window_size:]
                        self._get_buffer(p, 'reward')[:curr_len-window_size]=self._get_buffer(p, 'reward')[window_size:]
                        self._get_buffer(p, 'done')[:curr_len-window_size]=self._get_buffer(p, 'done')[window_size:]
                        self.write_indices[p] = curr_len-window_size
                        self.pool_lengths[p] = curr_len-window_size
                        if self.PR:
                            if self.PPO:
                                self._get_buffer(p, 'ratio')[:curr_len-window_size]=self._get_buffer(p, 'ratio')[window_size:]
                                self._get_buffer(p, 'TD')[:curr_len-window_size]=self._get_buffer(p, 'TD')[window_size:]
                            else:
                                self._get_buffer(p, 'TD')[:curr_len-window_size]=self._get_buffer(p, 'TD')[window_size:]
                            if hasattr(self.prioritized_replay, 'sum_tree'):
                                self.prioritized_replay.rebuild()
                    else:
                        self._get_buffer(p, 'state')[:curr_len-1]=self._get_buffer(p, 'state')[1:]
                        self._get_buffer(p, 'action')[:curr_len-1]=self._get_buffer(p, 'action')[1:]
                        self._get_buffer(p, 'next_state')[:curr_len-1]=self._get_buffer(p, 'next_state')[1:]
                        self._get_buffer(p, 'reward')[:curr_len-1]=self._get_buffer(p, 'reward')[1:]
                        self._get_buffer(p, 'done')[:curr_len-1]=self._get_buffer(p, 'done')[1:]
                        self.write_indices[p] = curr_len-1
                        self.pool_lengths[p] = curr_len-1
                        if self.PR:
                            if self.PPO:
                                self._get_buffer(p, 'ratio')[:curr_len-1]=self._get_buffer(p, 'ratio')[1:]
                                self._get_buffer(p, 'TD')[:curr_len-1]=self._get_buffer(p, 'TD')[1:]
                            else:
                                self._get_buffer(p, 'TD')[:curr_len-1]=self._get_buffer(p, 'TD')[1:]
                            if hasattr(self.prioritized_replay, 'sum_tree'):
                                self.prioritized_replay.rebuild()
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
    
    
    def select_action(self,s,i=None,p=None):
        if self.MARL!=True:
            if self.pool_network:
                output=self.action(s,p)
            else:
                output=self.action(s)
        else:
            if self.pool_network:
                output=self.action(s,i)
            else:
                output=self.action(s,i,p)
        if type(self.policy)==list:
            policy=self.policy[p]
        else:
            policy=self.policy
        if type(self.noise)==list:
            noise=self.noise[p]
        else:
            noise=self.noise
        if policy!=None:
            if self.IRL!=True:
                output=output.numpy()
            else:
                output=output[1].numpy()
            output=np.squeeze(output, axis=0)
            if isinstance(policy, rl.SoftmaxPolicy):
                a=policy.select_action(len(output), output)
            elif isinstance(policy, rl.EpsGreedyQPolicy):
                a=policy.select_action(output)
            elif isinstance(policy, rl.AdaptiveEpsGreedyPolicy):
                a=policy.select_action(output, self.step_counter)
            elif isinstance(policy, rl.GreedyQPolicy):
                a=policy.select_action(output)
            elif isinstance(policy, rl.BoltzmannQPolicy):
                a=policy.select_action(output)
            elif isinstance(policy, rl.MaxBoltzmannQPolicy):
                a=policy.select_action(output)
            elif isinstance(policy, rl.BoltzmannGumbelQPolicy):
                a=policy.select_action(output, self.step_counter)
        elif noise!=None:
            if self.IRL!=True:
                a=(output+noise.sample()).numpy()
            else:
                a=(output[1]+noise.sample()).numpy()
        else:
            a=output
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
        p = weights / (torch.sum(weights))
        ess = 1.0 / (torch.sum(p * p))
        return float(ess)


    def adjust_window_size(self, p=None, scale=1.0, ema=None):
        if self.pool_network==True:
            if ema is None:
                if self.PPO:
                    scores = self.lambda_ * self.TD_list[p] + (1.0-self.lambda_) * torch.abs(self.ratio_list[p] - 1.0)
                    weights = scores + 1e-7
                else:
                    weights = self.TD_list[p] + 1e-7
    
                if not self.parallel_store_and_training:
                    ess = self.compute_ess_from_weights(weights)
                elif self.end_flag:
                    ess = self.compute_ess_from_weights(weights)
                ema = ess
            else:
                ema = ema[p]
            
            ess = self.ess_[p]
        else:
            if ema is None:
                if self.PPO:
                    scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * torch.abs(self.prioritized_replay.ratio - 1.0)
                    weights = scores + 1e-7
                else:
                    weights = self.prioritized_replay.TD + 1e-7
                
                ess = self.compute_ess_from_weights(weights)
                
                ema = ess
            ess = self.ess_
            
        if not self.parallel_store_and_training:
            window_size = int((1.0 - ema / ess) * scale * len(weights))
        elif self.end_flag:
            if self.batch_counter != self.num_updates:
                window_size = int((1.0 - ema / (ess * self.num_updates / (self.num_updates - self.batch_counter))) * scale * len(weights))
            else:
                window_size = int((1.0 - ema / (ess * self.num_updates / (self.num_updates - (self.batch_counter - 1)))) * scale * len(weights))
         
        if window_size > 0:
            window_size = np.clip(window_size, 1, len(weights) - 1)
            return window_size
    
    
    def adjust_batch(self, batch_params, ema):
        if not hasattr(self, 'original_batch'):
            self.original_batch = self.batch
        if self.processes_her==None and self.processes_pr==None:
            buf_len = len(self.state_pool)
        else:
            buf_len = len(self.state_pool[7])
        if batch_params['min'] is None:
            cur_batch = self.batch
            batch_params['min'] = max(1, cur_batch // 2)
        if batch_params['max'] is None:
            batch_params['max'] = max(1, buf_len)
            
        if self.parallel_store_and_training:
            ess = self.ess.value
        else:
            ess = self.ess
            
        if not self.parallel_store_and_training:
            batch = int(round(self.batch * ema / ess * float(batch_params['scale'])))
        elif self.end_flag:
            if self.batch_counter != self.num_updates:
                batch = int(round(self.batch * ema / (ess * self.num_updates / (self.num_updates - self.batch_counter)) * float(batch_params['scale'])))
            else:
                batch = int(round(self.batch * ema / (ess * self.num_updates / (self.num_updates - (self.batch_counter - 1))) * float(batch_params['scale'])))
        batch = int(np.clip(batch, batch_params['min'], batch_params['max']))
        
        if batch_params['align'] is None:
            batch_params['align'] = self.batch
        new_batch = batch_params['align'] * (batch // batch_params['align'])
        self.batch = new_batch
    
    
    def adjust_alpha(self, alpha_params, ema=None, target=None, GNS=False):
        if not hasattr(self, 'original_alpha'):
            self.original_alpha = self.alpha
        if ema is None and not hasattr(self, 'ema_alpha'):
            self.ema_alpha = None
        smooth = alpha_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_alpha, smooth)
        if self.parallel_store_and_training:
            ess = self.ess.value
        else:
            ess = self.ess
        if not GNS:
            if not self.parallel_store_and_training:
                target_alpha = self.alpha + alpha_params['rate'] * (ema / ess - 1.0)
            elif self.end_flag:
                if self.batch_counter != self.num_updates:
                    target_alpha = self.alpha + alpha_params['rate'] * (ema / (ess * self.num_updates / (self.num_updates - self.batch_counter)) - 1.0)
                else:
                    target_alpha = self.alpha + alpha_params['rate'] * (ema / (ess * self.num_updates / (self.num_updates - (self.batch_counter - 1)) - 1.0))
        else:
            target_alpha = self.alpha + alpha_params['rate'] * (target - ema) / target
        alpha = np.clip(target_alpha, alpha_params['min'], alpha_params['max'])
        self.alpha = float(alpha)
    
    
    def adjust_eps(self, eps_params, eps, ema=None, target=None, GNS=False):
        if ema is None and not hasattr(self, 'ema_eps'):
            self.ema_eps = None
        smooth = eps_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_eps, smooth)
        if not GNS:
            target_eps = eps + eps_params['rate'] * (self.ess - ema) / self.ess
        else:
            target_eps = eps + eps_params['rate'] * (ema / target - 1.0)
        eps = np.clip(target_eps, eps_params['min'], eps_params['max'])
        return float(eps)


    def adjust_tau(self, tau_params, ema=None, target=None, GNS=False): 
        if not hasattr(self, 'original_tau'):
            self.original_tau = self.tau
        if ema is None and not hasattr(self, 'ema_tau'):
            self.ema_tau = None
        smooth = tau_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_tau, smooth)
        tau = self.tau
        if not GNS:
            target_tau = tau + tau_params['rate'] * (ema / self.ess - 1.0)
        else:
            target_tau = tau + tau_params['rate'] * (target - ema) / target
        tau = np.clip(target_tau, tau_params['min'], tau_params['max'])
        self.tau = float(tau)
    
    
    def adjust_gamma(self, gamma_params, ema=None, target=None, GNS=False): 
        if not hasattr(self, 'original_gamma'):
            self.original_gamma = self.gamma
        if ema is None and not hasattr(self, 'ema_gamma'):
            self.ema_gamma = None
        smooth = gamma_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_gamma, smooth)
        gamma = self.gamma
        if not GNS:
            target_gamma = gamma + gamma_params['rate'] * (ema / self.ess - 1.0)
        else:
            target_gamma = gamma + gamma_params['rate'] * (target - ema) / target
        gamma = np.clip(target_gamma, gamma_params['min'], gamma_params['max'])
        self.gamma = float(gamma)
        
        
    def adjust_num_store(self, store_params):  
        if not hasattr(self, 'original_num_store'):
            self.original_num_store = self.num_store
        scale = (1.0 - len(self.prioritized_replay.TD) / self.pool_size)
        if self.parallel_store_and_training:
            ess = self.ess.value
        else:
            ess = self.ess
        if scale > 0:
            if not self.parallel_store_and_training:
                num_store = store_params['scale'] * ess / self._ess * self.num_store * scale
            elif self.end_flag:
                if self.batch_counter != self.num_updates:
                    num_store = store_params['scale'] * (ess * self.num_updates / (self.num_updates - self.batch_counter)) / self._ess * self.num_store.value * scale
                else:
                    num_store = store_params['scale'] * (ess * self.num_updates / (self.num_updates - (self.batch_counter - 1))) / self._ess * self.num_store.value * scale
        else:
            if not self.parallel_store_and_training:
                num_store = store_params['scale'] * ess / self._ess * self.num_store
            elif self.end_flag:
                if self.batch_counter != self.num_updates:
                    num_store = store_params['scale'] * (ess * self.num_updates / (self.num_updates - self.batch_counter)) / self._ess * self.num_store.value
                else:
                    num_store = store_params['scale'] * (ess * self.num_updates / (self.num_updates - (self.batch_counter - 1))) / self._ess * self.num_store.value
        num_store = np.clip(num_store, store_params['min'], store_params['max'])
        if not self.parallel_store_and_training:
            self.num_store = int(max(store_params['min'], num_store))
        else:
            self.num_store.value = int(max(store_params['min'], num_store))
    
    
    def adjust_clip(self, clip_params, ema=None, target=None, GNS=False):  
        if not hasattr(self, 'original_clip'):
            self.original_clip = self.clip
        if ema is None and not hasattr(self, 'ema_clip'):
            self.ema_clip = None
        smooth = clip_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_clip, smooth)
        clip = self.clip
        if not GNS:
            target_clip = clip + clip_params['rate'] * (ema / self.ess - 1.0)
        else:
            target_clip = clip + clip_params['rate'] * (target - ema) / target
        clip = np.clip(target_clip, clip_params['min'], clip_params['max'])
        self.clip = float(clip)
        
        
    def adjust_beta(self, beta_params, ema=None, target=None, GNS=False):
        if not hasattr(self, 'original_beta'):
            self.original_beta = self.beta
        if ema is None and not hasattr(self, 'ema_beta'):
            self.ema_beta = None
        smooth = beta_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_beta, smooth)
        beta = self.beta
        if not GNS:
            target_beta = beta + beta_params['rate'] * (self.ess - ema) / self.ess
        else:
            target_beta = beta + beta_params['rate'] * (ema / target - 1.0)
        beta = np.clip(target_beta, beta_params['min'], beta_params['max'])
        self.beta = float(beta)
    
    
    def adjust_num_updates(self, num_updates_params, ema=None):
        if not hasattr(self, 'original_num_updates'):
            self.original_num_updates = self.num_updates
        if ema is None and not hasattr(self, 'ema_num_updates'):
            self.ema_num_updates = None
        smooth = num_updates_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_ess, smooth)
        target_num_updates = num_updates_params['scale'] * ema / self.ess * self.num_updates
        num_updates = np.clip(target_num_updates, num_updates_params['min'], num_updates_params['max'])
        if int(num_updates) <= self.batch_counter:
            return
        else:
            self.num_updates = int(num_updates)
    
    
    def compute_ess(self, ema_ess, smooth):
        if self.PPO:
            scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * np.abs(self.prioritized_replay.ratio - 1.0)
            weights = scores + 1e-7
        else:
            weights = self.prioritized_replay.TD + 1e-7
            
        p = weights / (torch.sum(weights))
        ess = 1.0 / (torch.sum(p * p))
        
        if ema_ess is None:
            ema = ess
        else:
            ema = smooth * ess + (1.0 - smooth) * ema_ess
            
        return float(ema)
    
    
    def adjust_batch_size(self, smooth=0.2, batch_params=None, target_ess=None, alpha_params=None, eps_params=None, tau_params=None, gamma_params=None, store_params=None, clip_params=None, beta_params=None, num_updates_params=None):
        if not hasattr(self, 'ema_ess'):
            self.ema_ess = None
        
        ema = self.compute_ess(self.ema_ess, smooth)
        
        if batch_params is not None:
            self.adjust_batch(batch_params, ema, target_ess)
        
        if alpha_params is not None and target_ess is not None:
            self.adjust_alpha(alpha_params, ema, target_ess)
    
        if eps_params is not None and target_ess is not None:
            if type(self.policy) == list:
                if not hasattr(self, 'original_eps'):
                    self.original_eps = [None for _ in range(len(self.policy))]
                for i, policy in enumerate(self.policy):
                    if hasattr(policy, 'eps'):
                        policy.eps = self.adjust_eps(eps_params, policy.eps, ema, target_ess)
                        if self.original_eps[i] is None:
                            self.original_eps[i] = policy.eps
            else:
                if not hasattr(self, 'original_eps'):
                    self.original_eps = None
                if hasattr(self.policy, 'eps'):
                    self.policy.eps = self.adjust_eps(eps_params, self.policy.eps, ema, target_ess)
                    if self.original_eps is None:
                        self.original_eps = self.policy.eps
        
        if tau_params is not None and target_ess is not None:
            self.adjust_tau(tau_params, ema, target_ess)
            
        if gamma_params is not None and target_ess is not None:
            self.adjust_gamma(gamma_params, ema, target_ess)
        
        if store_params is not None and target_ess is not None:
            self.adjust_num_store(store_params, ema, target_ess)
        
        if clip_params is not None and target_ess is not None:
            self.adjust_clip(clip_params, ema, target_ess)
        
        if beta_params is not None and target_ess is not None:
            self.adjust_beta(beta_params, ema, target_ess)
        
        if num_updates_params is not None and target_ess is not None:
            self.adjust_num_updates(num_updates_params, ema, target_ess)
    
    
    def estimate_gradient_variance(self, batch_size, num_samples, ema_noise, smooth):
        grads = []
        optimizer = torch.optim.SGD(self.param, lr=0.01)
    
        for _ in range(num_samples):
            idx = np.random.choice(self.state_pool.shape[0], size=batch_size, replace=False)
            if self.processes_her==None and self.processes_pr==None:
                s=self.state_pool[idx]
                a=self.action_pool[idx]
                next_s=self.next_state_pool[idx]
                r=self.reward_pool[idx]
                d=self.done_pool[idx]
            else:
                s=self.state_pool[7][idx]
                a=self.action_pool[7][idx]
                next_s=self.next_state_pool[7][idx]
                r=self.reward_pool[7][idx]
                d=self.done_pool[7][idx]
            optimizer.zero_grad()
            loss = self.__call__(s, a, next_s, r, d)
            loss.backward()
            grads_ = [p.grad.view(-1) for group in self.param for p in group if p.grad is not None]
            grad_flat = torch.cat(grads_)
            grads.append(grad_flat.clone())
            optimizer.zero_grad()
    
        grads = torch.stack(grads)
        mean_grad = grads.mean(dim=0)
        variance = ((grads - mean_grad) ** 2).mean().item()
        
        if ema_noise is None:
            ema_noise = variance
        else:
            ema_noise = smooth * variance + (1 - smooth) * ema_noise
            
        return ema_noise
    
    
    def adabatch(self, num_samples, target_noise=1e-3, smooth=0.2, batch_params=None, alpha_params=None, eps_params=None, tau_params=None, gamma_params=None, clip_params=None, beta_params=None):
        if not hasattr(self, 'ema_noise'):
            self.ema_noise = None
        
        ema_noise = self.estimate_gradient_variance(self.batch, num_samples, self.ema_noise, smooth)
        
        if batch_params is not None:
            self.adjust_batch(batch_params, ema_noise, target_noise)
        
        if alpha_params is not None:
            self.adjust_alpha(alpha_params, ema_noise, target_noise, True)
    
        if eps_params is not None:
            if type(self.policy) == list:
                if not hasattr(self, 'original_eps'):
                    self.original_eps = [None for _ in range(len(self.policy))]
                for i, policy in enumerate(self.policy):
                    if hasattr(policy, 'eps'):
                        policy.eps = self.adjust_eps(eps_params, policy.eps, ema_noise, target_noise, True)
                        if self.original_eps[i] is None:
                            self.original_eps[i] = policy.eps
            else:
                if not hasattr(self, 'original_eps'):
                    self.original_eps = None
                if hasattr(self.policy, 'eps'):
                    self.policy.eps = self.adjust_eps(eps_params, self.policy.eps, ema_noise, target_noise, True)
                    if self.original_eps is None:
                        self.original_eps = self.policy.eps
        
        if tau_params is not None:
            self.adjust_tau(tau_params, ema_noise, target_noise, True)
            
        if gamma_params is not None:
            self.adjust_gamma(gamma_params, ema_noise, target_noise, True)
        
        if clip_params is not None:
            self.adjust_clip(clip_params, ema_noise, target_noise, True)
        
        if beta_params is not None:
            self.adjust_beta(beta_params, ema_noise, target_noise, True)
    
    
    def initialize_adjusting(self):
        if hasattr(self, 'original_alpha'):
            self.alpha = self.original_alpha
            self.ema_alpha = None
        if hasattr(self, 'original_eps'):
            if type(self.policy) == list:
                for i, policy in enumerate(self.policy):
                    policy.eps = self.original_eps[i]
            else:
                self.policy.eps = self.original_eps
            self.ema_eps = None
        if hasattr(self, 'original_tau'):
            self.tau = self.original_tau
            self.ema_tau = None
        if hasattr(self, 'original_gamma'):
            self.gamma.assign(self.original_gamma)
            self.ema_gamma = None
        if hasattr(self, 'original_num_store'):
            if self.parallel_store_and_training:
                if self.num_store.value != self.original_num_store:
                    self.num_store.value = self.original_num_store
            else:
                self.num_store = self.original_num_store 
        if hasattr(self, 'original_clip'):
            self.clip.assign(self.original_clip)
            self.ema_clip = None
        if hasattr(self, 'original_beta'):
            self.beta.assign(self.original_beta)
            self.ema_beta = None
    
    
    def adjust(self, target_ess=None, target_noise=None, num_samples=None, smooth=0.2, batch_params=None, alpha_params=None, eps_params=None, tau_params=None, gamma_params=None, store_params=None, beta_params=None, clip_params=None):
        if target_noise is None:
            self.adjust_batch_size(smooth, batch_params, target_ess, alpha_params, eps_params, tau_params, gamma_params, store_params, clip_params, beta_params)
        else:
            self.adabatch(num_samples, target_noise, smooth, batch_params, alpha_params, eps_params, tau_params, gamma_params, clip_params, beta_params)
    
    
    def data_func(self, state_pool, action_pool, next_state_pool, reward_pool, done_pool):
        if self.parallel_store_and_training:
            done_length=len(done_pool)
            TD_list=[]
            ratio_list=[]
            self.length_list=[]
            for p in range(self.processes):
                TD_list.append(self._get_buffer(p, 'TD'))
                self.length_list.append(len(TD_list[p]))
                if self.PPO:
                    ratio_list.append(self._get_buffer(p, 'ratio')[:self.length_list[p]])
            TD=np.concat(TD_list, axis=0)
            np.frombuffer(self.shared_TD.get_obj(), dtype=np.float32)[:len(TD)]=TD
            if self.PPO:
                ratio=np.concat(ratio_list, axis=0)
                np.frombuffer(self.shared_ratio.get_obj(), dtype=np.float32)[:len(ratio)]=ratio
            length=min(done_length,len(TD))
            state_pool=state_pool[:length]
            action_pool=action_pool[:length]
            next_state_pool=next_state_pool[:length]
            reward_pool=reward_pool[:length]
            done_pool=done_pool[:length]
            self.prioritized_replay.TD=np.frombuffer(self.shared_TD.get_obj(), dtype=np.float32)[:length]
            if self.PPO:
                self.prioritized_replay.ratio=np.frombuffer(self.shared_ratio.get_obj(), dtype=np.float32)[:length]
            self.length_list[-1]=self.length_list[-1]-(len(TD)-len(self.prioritized_replay.TD))
        if self.PR:
            s,a,next_s,r,d=self.prioritized_replay.sample(state_pool,action_pool,next_state_pool,reward_pool,done_pool,self.lambda_,self.alpha,self.batch)
        elif self.HER:
            s = []
            a = []
            next_s = []
            r = []
            d = []
            for _ in range(self.batch):
                step_state = np.random.randint(0, len(state_pool)-1)
                step_goal = np.random.randint(step_state+1, step_state+np.argmax(done_pool[step_state+1:])+2)
                state = state_pool[step_state]
                next_state = next_state_pool[step_state]
                action = action_pool[step_state]
                goal = state_pool[step_goal]
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
        elif self.TRL:
            s_i_list = []
            s_k_list = []
            s_j_list = []
            a_i_list = []
            a_k_list = []
            r_i_list = []
            r_k_list = []
            d_i_list = []
            d_k_list = []
            
            done_indices = np.where(self.done_pool == 1.0)[0]

            for _ in range(self.batch):
                traj_end = np.random.choice(done_indices)
                
                possible_starts = np.where(self.done_pool[:traj_end] == 1.0)[0]
                traj_start = 0 if len(possible_starts) == 0 else possible_starts[-1] + 1
                    
                indices = np.sort(np.random.choice(np.arange(traj_start, traj_end + 1), 3, replace=False))
                idx_i, idx_k, idx_j = indices[0], indices[1], indices[2]
                
                s_i_list.append(state_pool[idx_i])
                s_k_list.append(next_state_pool[idx_k])
                s_j_list.append(next_state_pool[idx_j])
                a_i_list.append(action_pool[idx_i])
                a_k_list.append(action_pool[idx_k])
                r_i_list.append(reward_pool[idx_i])
                r_k_list.append(reward_pool[idx_k])
                d_i_list.append(done_pool[idx_i])
                d_k_list.append(done_pool[idx_k])
                
            s = np.stack((np.array(s_i_list), np.array(s_k_list)), axis=1)
            a = np.stack((np.array(a_i_list), np.array(a_k_list)), axis=1)
            next_s = np.stack((np.array(s_k_list), np.array(s_j_list)), axis=1)
            r = np.stack((np.array(r_i_list), np.array(r_k_list)), axis=1)
            d = np.stack((np.array(d_i_list), np.array(d_k_list)), axis=1)
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
    
    
    def clear_pool(self):
        for p in range(self.processes):
            if p==0:
                self._get_buffer(p, 'TD')[:self.length_list[p]]=self.prioritized_replay.TD[0:self.length_list[p]]
            else:
                index1=0
                index2=0
                for i in range(p):
                    index1+=self.length_list[i]
                index2=index1+self.length_list[i]
                self._get_buffer(p, 'TD')[:self.length_list[p]]=self.prioritized_replay.TD[index1-1:index2]
            if self.PPO:
                if p==0:
                    self._get_buffer(p, 'ratio')[:self.length_list[p]]=self.prioritized_replay.ratio[0:self.length_list[p]]
                else:
                    index1=0
                    index2=0
                    for i in range(p):
                        index1+=self.length_list[i]
                    index2=index1+self.length_list[p]
                    self._get_buffer(p, 'ratio')[:self.length_list[p]]=self.prioritized_replay.ratio[index1-1:index2]
        for p in range(self.processes):
            curr_len = self.pool_lengths[p]
            if curr_len==math.ceil(self.pool_size/self.processes):
                self.lock_list[p].acquire()
                if self.clearing_freq!=None:
                    self.store_counter[p]+=1
                    if self.store_counter[p]%self.clearing_freq==0:
                        self._get_buffer(p, 'state')[:curr_len-self.window_size_]=self._get_buffer(p, 'state')[self.window_size_:]
                        self._get_buffer(p, 'action')[:curr_len-self.window_size_]=self._get_buffer(p, 'action')[self.window_size_:]
                        self._get_buffer(p, 'next_state')[:curr_len-self.window_size_]=self._get_buffer(p, 'next_state')[self.window_size_:]
                        self._get_buffer(p, 'reward')[:curr_len-self.window_size_]=self._get_buffer(p, 'reward')[self.window_size_:]
                        self._get_buffer(p, 'done')[:curr_len-self.window_size_]=self._get_buffer(p, 'done')[self.window_size_:]
                        self.write_indices[p] = curr_len-self.window_size_
                        self.pool_lengths[p] = curr_len-self.window_size_
                        if self.PR:
                            if self.PPO:
                                self._get_buffer(p, 'ratio')[:curr_len-self.window_size_]=self._get_buffer(p, 'ratio')[self.window_size_:]
                                self._get_buffer(p, 'TD')[:curr_len-self.window_size_]=self._get_buffer(p, 'TD')[self.window_size_:]
                            else:
                                self._get_buffer(p, 'TD')[:curr_len-self.window_size_]=self._get_buffer(p, 'TD')[self.window_size_:]
                            if hasattr(self.prioritized_replay, 'sum_tree'):
                                self.prioritized_replay.rebuild()
                if type(self.window_size)!=int:
                    window_size=int(self.window_size(p))
                else:
                    window_size=self.window_size
                if window_size!=None:
                    self._get_buffer(p, 'state')[:curr_len-window_size]=self._get_buffer(p, 'state')[window_size:]
                    self._get_buffer(p, 'action')[:curr_len-window_size]=self._get_buffer(p, 'action')[window_size:]
                    self._get_buffer(p, 'next_state')[:curr_len-window_size]=self._get_buffer(p, 'next_state')[window_size:]
                    self._get_buffer(p, 'reward')[:curr_len-window_size]=self._get_buffer(p, 'reward')[window_size:]
                    self._get_buffer(p, 'done')[:curr_len-window_size]=self._get_buffer(p, 'done')[window_size:]
                    self.write_indices[p] = curr_len-window_size
                    self.pool_lengths[p] = curr_len-window_size
                    if self.PR:
                        self._get_buffer(p, 'TD')[:curr_len-window_size]=self._get_buffer(p, 'TD')[window_size:]
                        if self.PPO:
                            self._get_buffer(p, 'ratio')[:curr_len-window_size]=self._get_buffer(p, 'ratio')[window_size:]
                        if hasattr(self.prioritized_replay, 'sum_tree'):
                            self.prioritized_replay.rebuild()
                else:
                    self._get_buffer(p, 'state')[:curr_len-1]=self._get_buffer(p, 'state')[1:]
                    self._get_buffer(p, 'action')[:curr_len-1]=self._get_buffer(p, 'action')[1:]
                    self._get_buffer(p, 'next_state')[:curr_len-1]=self._get_buffer(p, 'next_state')[1:]
                    self._get_buffer(p, 'reward')[:curr_len-1]=self._get_buffer(p, 'reward')[1:]
                    self._get_buffer(p, 'done')[:curr_len-1]=self._get_buffer(p, 'done')[1:]
                    self.write_indices[p] = curr_len-1
                    self.pool_lengths[p] = curr_len-1
                    if self.PR:
                        self._get_buffer(p, 'TD')[:curr_len-1]=self._get_buffer(p, 'TD')[1:]
                        if self.PPO:
                            self._get_buffer(p, 'ratio')[:curr_len-1]=self._get_buffer(p, 'ratio')[1:]
                        if hasattr(self.prioritized_replay, 'sum_tree'):
                            self.prioritized_replay.rebuild()
                self.lock_list[p].release()
                
                
    def train1_pr(self, loss, batches):
        if self.parallel_store_and_training:
            self.update_pool()
        state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
        loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.optimizer)
        if self.parallel_store_and_training:
            np.frombuffer(self.shared_TD.get_obj(), dtype=np.float32)[self.prioritized_replay.index]=torch.abs(self.prioritized_replay.TD_[:self.prioritized_replay.batch])
            if self.PPO:
                np.frombuffer(self.shared_ratio.get_obj(), dtype=np.float32)[self.prioritized_replay.index]=self.prioritized_replay.ratio_[:self.prioritized_replay.batch]
            if hasattr(self.prioritized_replay, 'sum_tree'):
                self.prioritized_replay.update()
            self.clear_pool()
        else:
            self.prioritized_replay.update()
        self.batch_counter+=1
        if self.pool_network==True:
            if self.batch_counter%self.update_batches==0:
                self.update_param()
                if not hasattr(self,'window_size_func'):
                    if self.PPO:
                        window_size=self.window_size_ppo
                    else:
                        window_size=self.window_size_pr
                if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                    self._ess = self.compute_ess(None, None)
                for p in range(self.processes):
                    if self.parallel_store_and_training:
                        self.lock_list[p].acquire()
                    if hasattr(self,'window_size_func'):
                        window_size=int(self.window_size_func(p))
                        if self.PPO:
                            scores = self.lambda_ * self.TD_list[p] + (1.0-self.lambda_) * torch.abs(self.ratio_list[p] - 1.0)
                            weights = scores + 1e-7
                        else:
                            weights = self.TD_list[p] + 1e-7
                        p=weights/torch.sum(weights)
                        idx=np.random.choice(np.arange(len(self.state_pool_list[p])),size=[len(self.state_pool_list[p])-window_size],p=p.numpy(),replace=False)
                    if window_size!=None and len(self.state_pool_list[p])>window_size:
                        self.state_pool_list[p]=self.state_pool_list[p][idx]
                        self.action_pool_list[p]=self.action_pool_list[p][idx]
                        self.next_state_pool_list[p]=self.action_pool_list[p][idx]
                        self.reward_pool_list[p]=self.action_pool_list[p][idx]
                        self.done_pool_list[p]=self.action_pool_list[p][idx]
                        if self.PPO:
                            self.ratio_list[p]=self.ratio_list[p][idx]
                        self.TD_list[p]=self.TD_list[p][idx]
                        if not self.PPO:
                            weights = self.TD_list[p] + 1e-7
                            self.ess_[p] = self.compute_ess_from_weights(weights)
                    if self.parallel_store_and_training:
                        self.lock_list[p].release()
                if self.PPO:
                    self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                    self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                else:
                    self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                    if self.processes_her==None and self.processes_pr==None:
                        if self.parallel_store_and_training:
                            self.share_state_pool[7]=np.concatenate(self.state_pool_list)
                            self.share_action_pool[7]=np.concatenate(self.action_pool_list)
                            self.share_next_state_pool[7]=np.concatenate(self.next_state_pool_list)
                            self.share_reward_pool[7]=np.concatenate(self.reward_pool_list)
                            self.share_done_pool[7]=np.concatenate(self.done_pool_list)
                        else:
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
            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                self.adjust_func()
            if self.PPO and self.batch_counter%self.update_batches==0:
                return loss.detach().numpy()/batches
        return loss
    
    
    def train1(self):
        loss=0
        self.step_counter+=1
        self.end_flag=False
        batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
        if len(self.state_pool)%self.batch!=0:
            batches+=1
        if self.PR==True or self.HER==True or self.TRL==True:
            for j in range(batches):
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    break
                loss = self.train1_pr()
                if self.PPO and self.batch_counter%self.update_batches==0:
                    return loss
            if len(self.state_pool)%self.batch!=0:
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    return loss.detach().numpy()/batches
                loss = self.train1_pr()
                if self.PPO and self.batch_counter%self.update_batches==0:
                    return loss
        else:
            if self.parallel_store_and_training:
                state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.update_pool()
            else:
                state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.state_pool, self.action_pool, self.next_state_pool, self.reward_pool, self.done_pool
            if self.pool_network==True:
                if self.parallel_store_and_training:
                    length=len(done_pool)
                    train_ds=DataLoader((state_pool[:length],action_pool[:length],next_state_pool[:length],reward_pool[:length],done_pool[:length]),batch_size=self.batch)
                else:
                    train_ds=DataLoader((state_pool,action_pool,next_state_pool,reward_pool,done_pool),batch_size=self.batch)
            else:
                if self.num_updates!=None:
                    train_ds=DataLoader((state_pool,action_pool,next_state_pool,reward_pool,done_pool),batch_size=self.batch)
                else:
                    train_ds=DataLoader((state_pool,action_pool,next_state_pool,reward_pool,done_pool),batch_size=self.batch,shuffle=True)
            for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    break
                loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.optimizer)
                self.batch_counter+=1
                if self.pool_network==True:
                    if self.batch_counter%self.update_batches==0:
                        self.update_param()
                        if self.PPO:
                            for p in range(self.processes):
                                if self.parallel_store_and_training:
                                    self.lock_list[p].acquire()
                                self.write_indices[p]=0
                                self.pool_lengths[p]=0
                                if self.parallel_store_and_training:
                                    self.lock_list[p].release()
                    if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                        self.adjust_func()
                    if self.PPO and self.batch_counter%self.update_batches==0:
                        break
        if self.update_steps!=None:
            if self.step_counter%self.update_steps==0:
                self.update_param()
                if self.PR:
                    if not hasattr(self,'window_size_func'):
                        if self.PPO:
                            window_size=self.window_size_ppo
                        else:
                            window_size=self.window_size_pr
                    if hasattr(self, 'adjust_func'):
                        self._ess = self.compute_ess(None, None)
                    if hasattr(self,'window_size_func'):
                        window_size=int(self.window_size_func())
                        if self.PPO:
                            scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * torch.abs(self.prioritized_replay.ratio - 1.0)
                            weights = scores + 1e-7
                        else:
                            weights = self.prioritized_replay.TD + 1e-7
                        p=weights/torch.sum(weights)
                        idx=np.random.choice(np.arange(len(self.state_pool)),size=[len(self.state_pool)-window_size],p=p.numpy(),replace=False)
                    if window_size!=None and len(self.state_pool)>window_size:
                        self.state_pool=self.state_pool[idx]
                        self.action_pool=self.action_pool[idx]
                        self.next_state_pool=self.action_pool[idx]
                        self.reward_pool=self.action_pool[idx]
                        self.done_pool=self.action_pool[idx]
                        if self.PPO:
                            self.prioritized_replay.ratio=self.prioritized_replay.ratio[idx]
                        self.prioritized_replay.TD=self.prioritized_replay.TD[idx]
                        if not self.PPO:
                            weights = self.prioritized_replay.TD + 1e-7
                            self.ess_ = self.compute_ess_from_weights(weights)
                elif self.PPO:
                    self.state_pool=None
                    self.action_pool=None
                    self.next_state_pool=None
                    self.reward_pool=None
                    self.done_pool=None
            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                self.adjust_func()
            if self.PPO and self.step_counter%self.update_steps==0:
                return loss.detach().numpy()/batches
        elif self.pool_network==False:
            self.update_param()
        return loss.detach().numpy()/batches
    
    
    def train2(self):
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
                        self.prioritized_replay.TD=np.append(self.prioritized_replay.TD,np.max(self.prioritized_replay.TD))
                else:
                    if len(self.state_pool)>1:
                        self.prioritized_replay.TD=np.append(self.prioritized_replay.TD,np.max(self.prioritized_replay.TD))
                if hasattr(self,'window_size_func'):
                    if not hasattr(self,'ess_'):
                        self.ess_ = None
                    if self.PPO:
                        scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * torch.abs(self.prioritized_replay.ratio - 1.0)
                        weights = scores + 1e-7
                    else:
                        weights = self.prioritized_replay.TD + 1e-7
                    self.ess_ = self.compute_ess_from_weights(weights)
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
            if (self.num_steps==None and done) or (self.num_steps!=None and done_):
                if len(self.state_pool)<self.batch:
                    s=self.env_(initial=True)
                    continue
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
            loss=self.train1()
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
                return loss
            s=next_s
            if (self.num_steps!=None and counter%self.num_steps==0) or (self.num_steps!=None and done):
                s=next_s_
   
    
    def modify_ratio_TD(self):
        if self.PR==True:
            for p in range(self.processes):
                if self.prioritized_replay.ratio is not None:
                    if p==0:
                        self._get_buffer(p, 'ratio')[:self.pool_lengths[p]]=self.prioritized_replay.ratio[:self.pool_lengths[p]]
                    else:
                        index1=0
                        index2=0
                        for i in range(p):
                            index1+=self.pool_lengths[i]
                        index2=index1+self.pool_lengths[p]
                        self._get_buffer(p, 'ratio')[:self.pool_lengths[p]]=self.prioritized_replay.ratio[index1-1:index2]
                if self.prioritized_replay.TD is not None:
                    if p==0:
                        self._get_buffer(p, 'TD')[:self.pool_lengths[p]]=self.prioritized_replay.TD[:self.pool_lengths[p]]
                    else:
                        index1=0
                        index2=0
                        for i in range(p):
                            index1+=self.pool_lengths[i]
                        index2=index1+self.pool_lengths[p]
                        self._get_buffer(p, 'TD')[:self.pool_lengths[p]]=self.prioritized_replay.TD[index1-1:index2]
        return
    
    
    def modify_TD(self):
        if self.PR==True:
            for p in range(self.processes):
                if self.prioritized_replay.TD is not None:
                    if p==0:
                        self._get_buffer(p, 'TD')[:self.pool_lengths[p]]=self.prioritized_replay.TD[:self.pool_lengths[p]]
                    else:
                        index1=0
                        index2=0
                        for i in range(p):
                            index1+=self.pool_lengths[i]
                        index2=index1+self.pool_lengths[p]
                        self._get_buffer(p, 'TD')[:self.pool_lengths[p]]=self.prioritized_replay.TD[index1-1:index2]
        return
    
    
    def store_in_parallel(self,p):
        self.reward[p]=0
        s=self.env_(initial=True,p=p)
        s=np.array(s)
        reward=0
        counter=0
        while True:
            if self.random or (self.PR!=True and self.HER!=True and self.TRL!=True):
                if self.write_indices[p] == 0:
                    p=p
                    self.inverse_len[p]=1
                else:
                    inverse_len=torch.tensor(self.inverse_len)
                    total_inverse=torch.sum(inverse_len)
                    prob=inverse_len/total_inverse
                    p=np.random.choice(self.processes,p=prob.numpy(),replace=False)
                    self.inverse_len[p]=1/self.pool_lengths[p]
            else:
                p=p
            s=np.expand_dims(s,axis=0)
            if self.MARL!=True:
                a=self.select_action(s,p=p)
            else:
                a=[]
                for i in len(s[0]):
                    s=np.expand_dims(s[0][i],axis=0)
                    a.append([self.select_action(s,i,p)])
                a=np.array(a)
            next_s,r,done=self.env_(a,p=p)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            if self.random or (self.PR!=True and self.HER!=True and self.TRL!=True):
                self.lock_list[p].acquire()
                if self.num_steps!=None:
                    if counter==0:
                        next_s_=next_s
                        done_=done
                    counter+=1
                    reward=r+reward
                    if counter%self.num_steps==0 or done:
                        self.pool(s,a,next_s,reward,done,p)
                        reward=0
                else:
                    self.pool(s,a,next_s,r,done,p)
                self.lock_list[p].release()
            else:
                if self.parallel_store_and_training:
                    self.lock_list[p].acquire()
                if self.num_steps!=None:
                    if counter==0:
                        next_s_=next_s
                        done_=done
                    counter+=1
                    reward=r+reward
                    if counter%self.num_steps==0 or done:
                        self.pool(s,a,next_s,reward,done,p)
                        reward=0
                else:
                    self.pool(s,a,next_s,r,done,p)
                if self.parallel_store_and_training:
                    self.lock_list[p].release()
            self.done_length[p]=len(self.done_pool_list[p])
            if self.MARL==True:
                r,done=self.reward_done_func_ma(r,done)
            self.reward[p]=r+self.reward[p]
            if (self.num_steps==None and done) or (self.num_steps!=None and done_):
                self.reward_list.append(self.reward[p])
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
                return
            s=next_s
            if (self.num_steps!=None and counter%self.num_steps==0) or (self.num_steps!=None and done):
                s=next_s_
    
    
    def prepare(self, p=None):
        process_list=[]
        if not self.parallel_store_and_training and self.PPO:
            self.modify_ratio_TD()
        elif not self.parallel_store_and_training:
            self.modify_TD()
        counter=0
        if hasattr(self, 'original_batch'):
            self.batch = self.original_batch
            if hasattr(self, 'ema_ess'):
                self.ema_ess = None
            else:
                self.ema_noise = None
        if hasattr(self, 'original_num_updates'):
            self.num_updates = self.original_num_updates
            self.ema_num_updates = None
            self.batch_counter = 0
        if self.parallel_store_and_training:
            num_store = self.num_store.value
        else:
            num_store = self.num_store
        while True:
            if self.parallel_store_and_training:
                self.store_in_parallel(p)
            else:
                for p in range(self.processes):
                    process=mp.Process(target=self.store_in_parallel,args=(p,))
                    process.start()
                    process_list.append(process)
                for process in process_list:
                    process.join()
            counter+=1
            if self.parallel_store_and_training and counter==num_store:
                break
            if not self.parallel_store_and_training:
                if self.state_pool is not None and len(self.state_pool)>=self.batch and counter<self.num_store:
                        continue
                state_pools = [self._get_buffer(p, 'state')[:self.pool_lengths[p]] for p in range(self.processes)]
                self.state_pool = np.concatenate(state_pools, axis=0)
                action_pools = [self._get_buffer(p, 'action')[:self.pool_lengths[p]] for p in range(self.processes)]
                self.action_pool = np.concatenate(action_pools, axis=0)
                next_state_pools = [self._get_buffer(p, 'next_state')[:self.pool_lengths[p]] for p in range(self.processes)]
                self.next_state_pool = np.concatenate(next_state_pools, axis=0)
                reward_pools = [self._get_buffer(p, 'reward')[:self.pool_lengths[p]] for p in range(self.processes)]
                self.reward_pool = np.concatenate(reward_pools, axis=0)
                done_pools = [self._get_buffer(p, 'done')[:self.pool_lengths[p]] for p in range(self.processes)]
                self.done_pool = np.concatenate(done_pools, axis=0)
                if counter<self.num_store and len(self.state_pool)<self.batch:
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
                if len(self.state_pool)>=self.batch:
                    break
        if not self.parallel_store_and_training:
            if hasattr(self,'window_size_func'):
                for p in range(self.processes):
                    if not hasattr(self,'ess_'):
                        self.ess_ = [None] * self.processes
                    if self.PPO:
                        scores = self.lambda_ * self.TD_list[p] + (1.0-self.lambda_) * torch.abs(self.ratio_list[p] - 1.0)
                        weights = scores + 1e-7
                    else:
                        weights = self.TD_list[p] + 1e-7
                    self.ess_[p] = self.compute_ess_from_weights(weights)
            self.initialize_adjusting()    
            if self.PR==True:
                if self.PPO:
                    ratio_list = [self._get_buffer(p, 'ratio')[:self.pool_lengths[p]] for p in range(self.processes)]
                    self.prioritized_replay.ratio = np.concatenate(ratio_list, axis=0)
                    TD_list = [self._get_buffer(p, 'TD')[:self.pool_lengths[p]] for p in range(self.processes)]
                    self.prioritized_replay.TD = np.concatenate(TD_list, axis=0)
                else:
                    TD_list = [self._get_buffer(p, 'TD')[:self.pool_lengths[p]] for p in range(self.processes)]
                    self.prioritized_replay.TD = np.concatenate(TD_list, axis=0)
                if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                    self.ess=self.compute_ess(None,None)
        else:
            self.end_flag_list[p]=True
    
    
    def update_pool(self):
        if not self.end_flag:
            if all(self.end_flag_list):
                self.end_flag=True
                for p in range(self.processes):
                    self.end_flag_list[p]=False
                    self.done_length[p]=0
            state_pools = []
            action_pools = []
            next_state_pools = []
            reward_pools = []
            done_pools = []
            for p in range(self.processes):
                curr_len = self.pool_lengths[p]
                state_pools.append(self._get_buffer(p, 'state')[:curr_len])
                action_pools.append(self._get_buffer(p, 'action')[:curr_len])
                next_state_pools.append(self._get_buffer(p, 'next_state')[:curr_len])
                reward_pools.append(self._get_buffer(p, 'reward')[:curr_len])
                done_pools.append(self._get_buffer(p, 'done')[:curr_len])
            state_pool = np.concatenate(state_pools, axis=0)
            action_pool = np.concatenate(action_pools, axis=0)
            next_state_pool = np.concatenate(next_state_pools, axis=0)
            reward_pool = np.concatenate(reward_pools, axis=0)
            done_pool = np.concatenate(done_pools, axis=0)
            if not self.PR and self.num_updates!=None:
                if len(done_pool)>=self.pool_size_:
                    idx=np.random.choice(done_pool.shape[0], size=self.pool_size_, replace=False)
                else:
                    idx=np.random.choice(done_pool.shape[0], size=done_pool.shape[0], replace=False)
                state_pool=state_pool[idx]
                action_pool=action_pool[idx]
                next_state_pool=next_state_pool[idx]
                reward_pool=reward_pool[idx]
                done_pool=done_pool[idx]
        if self.end_flag:
            if hasattr(self,'window_size_func'):
                for p in range(self.processes):
                    if not hasattr(self,'ess_'):
                        self.ess_ = [None] * self.processes
                    if self.PPO:
                        scores = self.lambda_ * self._get_buffer(p, 'TD')[:curr_len] + (1.0-self.lambda_) * torch.abs(self._get_buffer(p, 'ratio')[:curr_len] - 1.0)
                        weights = scores + 1e-7
                    else:
                        weights = self._get_buffer(p, 'TD')[:curr_len] + 1e-7
                    self.ess_[p] = self.compute_ess_from_weights(weights)
            self.initialize_adjusting()
            if self.PR==True:
                if hasattr(self, 'adjust_func') and len(done_pool)>=self.pool_size_:
                    self.ess.value=self.compute_ess(None,None)
                    
                    
    def _init_shared_experience_buffers(self, processes):
        s_elements = int(np.prod(self.state_shape))
        a_elements = int(np.prod(self.action_shape))

        self.shared_states = []
        self.shared_actions = []
        self.shared_next_states = []
        self.shared_rewards = []
        self.shared_dones = []
        self.shared_TDs = []
        self.shared_ratios = []

        for _ in range(processes):
            self.shared_states.append(mp.Array('f', self.max_exp_per_proc * s_elements))
            self.shared_next_states.append(mp.Array('f', self.max_exp_per_proc * s_elements))
            self.shared_actions.append(mp.Array('f', self.max_exp_per_proc * a_elements))
            self.shared_rewards.append(mp.Array('f', self.max_exp_per_proc))
            self.shared_dones.append(mp.Array('f', self.max_exp_per_proc))

            if self.PR:
                self.shared_TDs.append(mp.Array('f', self.max_exp_per_proc))
                if self.PPO:
                    self.shared_ratios.append(mp.Array('f', self.max_exp_per_proc))
    
    
    def _get_buffer(self, p, field):
        if field == 'state':
            arr = np.frombuffer(self.shared_states[p].get_obj(), dtype=np.float32)
            return arr.reshape((self.max_exp_per_proc,) + self.state_shape)
        if field == 'action':
            arr = np.frombuffer(self.shared_actions[p].get_obj(), dtype=np.float32)
            return arr.reshape((self.max_exp_per_proc,) + self.action_shape)
        if field == 'next_state':
            arr = np.frombuffer(self.shared_next_states[p].get_obj(), dtype=np.float32)
            return arr.reshape((self.max_exp_per_proc,) + self.next_state_shape)
        if field == 'reward':
            return np.frombuffer(self.shared_rewards[p].get_obj(), dtype=np.float32)
        if field == 'done':
            return np.frombuffer(self.shared_dones[p].get_obj(), dtype=np.float32)
        if field == 'TD':
            return np.frombuffer(self.shared_TDs[p].get_obj(), dtype=np.float32)
        if field == 'ratio':
            return np.frombuffer(self.shared_ratios[p].get_obj(), dtype=np.float32)
    
    
    def train(self, optimizer=None, episodes=None, pool_network=True, parallel_store_and_training=True, processes=None, num_store=1, processes_her=None, processes_pr=None, window_size=None, clearing_freq=None, window_size_=None, window_size_ppo=None, window_size_pr=None, random=False, save_data=True, p=None):
        self.avg_reward=None
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
        self.optimizer=optimizer  
        self.pool_network=pool_network
        if pool_network:
            manager=mp.Manager()
        self.parallel_store_and_training=parallel_store_and_training
        if parallel_store_and_training:
            if self.PR and self.PPO:
                self.shared_TD=mp.Array('f', self.pool_size)
                self.shared_ratio=mp.Array('f', self.pool_size)
            elif self.PR:
                self.shared_TD=mp.Array('f', self.pool_size)
            self.done_length=manager.list([0 for _ in range(processes)])
            self.ess=mp.Value('f',0)
            self.original_num_store=self.num_store
            self.num_store=mp.Value('i',self.num_store)
            self.ess_=manager.list([None for _ in range(processes)])
            self.end_flag_list=manager.list([False for _ in range(processes)])
        self.processes=processes
        self.num_store=num_store
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
            if self.state_shape is None:
                test_s = self.env_(initial=True, p=0)
                test_s = np.asarray(test_s)
                if getattr(self, 'MARL', False):
                    if test_s.ndim == 1:
                        test_s = test_s.reshape(1, -1)
                    self.state_shape = test_s.shape
                    num_agents = test_s.shape[0]
                    dummy_s = np.expand_dims(test_s, axis=0)
                    dummy_a_list = []
                    for i in range(num_agents):
                        agent_s = np.expand_dims(test_s[i], axis=0)
                        agent_a = self.select_action(agent_s, i=i, p=0)
                        dummy_a_list.append(np.asarray(agent_a).squeeze())
                    dummy_a = np.stack(dummy_a_list, axis=0)
                    
                else:
                    self.state_shape = test_s.shape
                    dummy_s = np.expand_dims(test_s, axis=0)
                    dummy_a = self.select_action(dummy_s, p=0)
                    dummy_a = np.asarray(dummy_a)
                self.action_shape = dummy_a.shape if dummy_a.ndim > 0 else (1,)
                self.next_state_shape = self.state_shape
            self.max_exp_per_proc = math.ceil(self.pool_size / self.processes * self.buffer_safety_factor)
            self._init_shared_experience_buffers(processes)
            if save_data:
                self.pool_lengths = manager.list(self.pool_lengths)
                self.write_indices = manager.list(self.write_indices)
                self.inverse_len=manager.list([0 for _ in range(processes)])
                if self.clearing_freq!=None:
                    self.store_counter=manager.list(self.store_counter)
            else:
                self.pool_lengths = manager.list([0 for _ in range(processes)])
                self.write_indices = manager.list([0 for _ in range(processes)])
                self.inverse_len=manager.list([0 for _ in range(processes)])
                if self.clearing_freq!=None:
                    self.store_counter=manager.list([0 for _ in range(processes)])
            self.reward=manager.list([0 for _ in range(processes)])
            if parallel_store_and_training or self.HER!=True or self.TRL!=True:
                self.lock_list=[manager.Lock() for _ in range(processes)]
            if self.PR==True:
                if self.PPO:
                    self.prioritized_replay.ratio=None
                    self.prioritized_replay.TD=None
                else:
                    self.prioritized_replay.TD=None
        if episodes!=None:
            for i in range(episodes):
                t1=time.time()
                if pool_network==True:
                    if parallel_store_and_training:
                        process_list=[]
                        for p in range(processes):
                            process=mp.Process(target=self.prepare,args=(p,))
                            process.start()
                            process_list.append(process)
                        while True:
                            if sum(self.done_length)>=self.batch:
                                break
                        self.train1()
                        for process in process_list:
                            process.join()
                    else:
                        self.prepare()
                        loss=self.train1()
                else:
                    loss=self.train2()
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
                        self.avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and self.avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('average reward:{0}'.format(self.avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if self.avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,self.avg_reward))
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
                    if parallel_store_and_training:
                        process_list=[]
                        for p in range(processes):
                            process=mp.Process(target=self.prepare,args=(p,))
                            process.start()
                            process_list.append(process)
                        while True:
                            if sum(self.done_length)>=self.batch:
                                break
                        self.train1()
                        for process in process_list:
                            process.join()
                    else:
                        self.prepare()
                        loss=self.train1()
                else:
                    loss=self.train2()
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
                        self.avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and self.avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('average reward:{0}'.format(self.avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if self.avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,self.avg_reward))
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
                if self.avg_reward!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.avg_reward))
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
                    self.avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.self.avg_reward==None or self.avg_reward>self.self.avg_reward:
                        self.save_param(path)
                        self.self.avg_reward=self.avg_reward
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
        if self.save_best_only==False:
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(path,'wb')
            else:
                if self.avg_reward!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.avg_reward))
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
                    self.avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.self.avg_reward==None or self.avg_reward>self.self.avg_reward:
                        self.save(path)
                        self.self.avg_reward=self.avg_reward
        return
    
    
    def save(self,path):
        output_file=open(path,'wb')
        pickle.dump(self,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        model=pickle.load(input_file)
        self.__dict__.update(model.__dict__)
        input_file.close()
        return