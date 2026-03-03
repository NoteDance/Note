import tensorflow as tf
from Note import nn
from tensorflow.python.util import nest
import multiprocessing as mp
from Note.RL import rl
from Note.RL.rl.prioritized_replay import pr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import statistics
import pickle
import os
import shutil
import time


class RL:
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
        self.buffer_safety_factor=1.5
        self.seed=7
        self.path=None
        self.save_freq=1
        self.save_freq_=None
        self.max_save_files=1
        self.best_avg_reward=None
        self.best_avg_reward_=None
        self.patience=None
        self.patience_counter=0
        self.save_top_k=1
        self.save_last=True
        self.save_param_only=False
        self.callbacks=[]
        self.stop_training=False
        self.info=dict()
        self.path_list=[]
        self.loss=None
        self.loss_list=[]
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def get_info(self):
        self.info['policy']=self.policy
        self.info['noise']=self.noise
        self.info['pool_size']=self.pool_size
        self.info['batch']=self.batch
        self.info['num_updates']=self.num_updates
        self.info['num_steps']=self.num_steps
        self.info['update_batches']=self.update_batches
        self.info['update_steps']=self.update_steps
        self.info['trial_count']=self.trial_count
        self.info['criterion']=self.criterion
        self.info['PPO']=self.PPO
        self.info['HER']=self.HER
        self.info['TRL']=self.TRL
        self.info['MARL']=self.MARL
        self.info['PR']=self.PR
        self.info['IRL']=self.IRL
        self.info['initial_ratio']=self.initial_ratio
        self.info['initial_TD']=self.initial_TD
        self.info['lambda_']=self.lambda_
        self.info['alpha']=self.alpha
        self.info['path']=self.path
        self.info['save_freq']=self.save_freq
        self.info['save_freq_']=self.save_freq_
        self.info['max_save_files']=self.max_save_files
        self.info['save_best_only']=self.save_best_only
        self.info['save_param_only']=self.save_param_only
        self.info['total_episode']=self.total_episode
        self.info['time']=self.time
        self.info['total_time']=self.total_time
        if self.info_flag==0:
            try:
                self.info['train_loss']=self.train_loss
                self.info['episodes']=self.episodes
                self.info['jit_compile']=self.jit_compile
                self.info['pool_network']=self.pool_network
                self.info['parallel_store_and_training']=self.parallel_store_and_training
                self.info['parallel_training_and_save']=self.parallel_training_and_save
                self.info['processes']=self.processes
                self.info['num_store']=self.num_store
                self.info['processes_her']=self.processes_her
                self.info['processes_pr']=self.processes_pr
                self.info['window_size']=self.window_size
                self.info['clearing_freq']=self.clearing_freq
                self.info['window_size_']=self.window_size_
                self.info['window_size_ppo']=self.window_size_ppo
                self.info['window_size_pr']=self.window_size_pr
                self.info['random']=self.random
                self.info['num_updates']=self.num_updates
                self.info['save_data']=self.save_data
                self.info['p']=self.p
                if type(self.optimizer)==list:
                    self.info['optimizer']=[tf.keras.optimizers.serialize(optimizer) for optimizer in self.optimzer]
                else:
                    self.info['optimizer']=tf.keras.optimizers.serialize(self.optimizer)
            except Exception:
                pass
        else:
            try:
                self.info['strategy']=self.strategy
                self.info['episodes']=self.episodes
                self.info['num_episodes']=self.num_episodes
                self.info['jit_compile']=self.jit_compile
                self.info['pool_network']=self.pool_network
                self.info['parallel_store_and_training']=self.parallel_store_and_training
                self.info['parallel_training_and_save']=self.parallel_training_and_save
                self.info['processes']=self.processes
                self.info['num_store']=self.num_store
                self.info['processes_her']=self.processes_her
                self.info['processes_pr']=self.processes_pr
                self.info['window_size']=self.window_size
                self.info['clearing_freq']=self.clearing_freq
                self.info['window_size_']=self.window_size_
                self.info['window_size_ppo']=self.window_size_ppo
                self.info['window_size_pr']=self.window_size_pr
                self.info['random']=self.random
                self.info['num_updates']=self.num_updates
                self.info['save_data']=self.save_data
                self.info['p']=self.p
                if type(self.optimizer)==list:
                    self.info['optimizer']=[tf.keras.optimizers.serialize(optimizer) for optimizer in self.optimzer]
                else:
                    self.info['optimizer']=tf.keras.optimizers.serialize(self.optimizer)
            except Exception:
                pass
        return self.info
    
    
    def set(self,policy=None,noise=None,pool_size=None,batch=None,num_updates=None,num_steps=None,update_batches=None,update_steps=None,trial_count=None,criterion=None,PPO=False,HER=False,TRL=False,MARL=False,PR=False,IRL=False,initial_ratio=1.0,initial_TD=7.,lambda_=0.5,alpha=0.7):
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
            if PPO:
                self.prioritized_replay.PPO=PPO
                self.prioritized_replay.ratio=self.initial_ratio
                self.prioritized_replay.TD=self.initial_TD
                self.prioritized_replay.ratio_=tf.Variable(tf.zeros([batch]))
                self.prioritized_replay.TD_=tf.Variable(tf.zeros([batch]))
                self.prioritized_replay.batch=tf.Variable(tf.zeros((),dtype=tf.int32))
            else:
                self.prioritized_replay.TD=self.initial_TD
                self.prioritized_replay.TD_=tf.Variable(tf.zeros([batch]))
                self.prioritized_replay.batch=tf.Variable(tf.zeros((),dtype=tf.int32))
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
                                self._get_buffer(p, 'ratio')=self._get_buffer(p, 'ratio')[self.window_size_:]
                                self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[self.window_size_:]
                            else:
                                self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[self.window_size_:]
                if curr_len>math.ceil(self.pool_size/self.processes):
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
                                self._get_buffer(p, 'ratio')=self._get_buffer(p, 'ratio')[window_size:]
                                self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[window_size:]
                            else:
                                self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[window_size:]
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
                                self._get_buffer(p, 'ratio')=self._get_buffer(p, 'ratio')[1:]
                                self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[1:]
                            else:
                                self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[1:]
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
    
    
    @tf.function(jit_compile=True)
    def forward(self,s,i):
        if self.MARL!=True:
            output=self.action(s)
        else:
            output=self.action(s,i)
        return output
    
    
    @tf.function
    def forward_(self,s,i):
        if self.MARL!=True:
            output=self.action(s)
        else:
            output=self.action(s,i)
        return output
    
    
    def select_action(self,s,i=None,p=None):
        if type(self.policy)==list:
            policy=self.policy[p]
        else:
            policy=self.policy
        if type(self.noise)==list:
            noise=self.noise[p]
        else:
            noise=self.noise
        if policy!=None or noise!=None:
            if self.jit_compile==True:
                output=self.forward(s,i)
            else:
                output=self.forward_(s,i)
        else:
            if self.MARL!=True:
                if self.pool_network:
                    a=self.action(s,p)
                else:
                    a=self.action(s)
            else:
                if self.pool_network:
                    a=self.action(s,i)
                else:
                    a=self.action(s,i,p)
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
    
    
    def check_early_stopping(self):
        if len(self.reward_list)>=self.trial_count:
            avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
            if self.best_avg_reward_==None:
                self.best_avg_reward_=avg_reward
            if avg_reward<self.best_avg_reward_:
                self.patience_counter += 1
            elif avg_reward>self.best_avg_reward:
                self.patience_counter = 0
                self.best_avg_reward_=avg_reward
        if self.patience_counter == self.patience:
            self.stop_training = True
    
    
    def compute_ess_from_weights(self, weights):
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        return float(ess)


    def adjust_window_size(self, p=None, scale=1.0, ema=None):
        if self.pool_network==True:
            if ema is None:
                if self.PPO:
                    scores = self.lambda_ * self.TD_list[p] + (1.0-self.lambda_) * tf.abs(self.ratio_list[p] - 1.0)
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
                    scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
                    weights = scores + 1e-7
                else:
                    weights = self.prioritized_replay.TD
                
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
            self.original_gamma = self.gamma.numpy()
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
        self.gamma.assign(gamma)
        
        
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
            self.original_clip = self.clip.numpy()
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
        self.clip.assign(clip)
        
        
    def adjust_beta(self, beta_params, ema=None, target=None, GNS=False):
        if not hasattr(self, 'original_beta'):
            self.original_beta = self.beta.numpy()
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
        self.beta.assign(beta)
        
        
    def adjust_num_updates(self, num_updates_params, ema=None):
        if not hasattr(self, 'original_num_updates'):
            self.original_num_updates = self.num_updates
        if ema is None and not hasattr(self, 'ema_num_updates'):
            self.ema_num_updates = None
        smooth = num_updates_params.get('smooth', 0.2)
        if ema is None:
            ema = self.compute_ess(self.ema_num_updates, smooth)
        target_num_updates = num_updates_params['scale'] * ema / self.ess * self.num_updates
        num_updates = np.clip(target_num_updates, num_updates_params['min'], num_updates_params['max'])
        if int(num_updates) <= self.batch_counter:
            return
        else:
            self.num_updates = int(num_updates)
    
    
    def compute_ess(self, ema_ess, smooth):
        if self.PPO:
            scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
            weights = scores + 1e-7
        else:
            weights = self.prioritized_replay.TD + 1e-7
            
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        
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
    
    
    @tf.function(jit_compile=True)
    def backward(self, s, a, next_s, r, d):
        with tf.GradientTape() as tape:
            loss = self.__call__(s, a, next_s, r, d)
        gradients = tape.gradient(loss, self.param)
        return gradients
    
    
    @tf.function
    def backward_(self, s, a, next_s, r, d):
        with tf.GradientTape() as tape:
            loss = self.__call__(s, a, next_s, r, d)
        gradients = tape.gradient(loss, self.param)
        return gradients
    
    
    def estimate_gradient_variance(self, batch_size, num_samples, ema_noise, smooth, jit_compile=True):
        grads = []
    
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
            if jit_compile==True:
                gradients = self.backward(s, a, next_s, r, d)
            else:
                gradients = self.backward_(s, a, next_s, r, d)
            grad_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients], axis=0)
            grads.append(grad_flat)
    
        grads = tf.stack(grads)
        mean_grad = tf.reduce_mean(grads, axis=0)
        variance = tf.reduce_mean((grads - mean_grad) ** 2)
        
        if ema_noise is None:
            ema_noise = variance
        else:
            ema_noise = smooth * variance + (1 - smooth) * ema_noise
            
        return ema_noise
    
    
    def adabatch(self, num_samples, target_noise=1e-3, smooth=0.2, batch_params=None, alpha_params=None, eps_params=None, tau_params=None, gamma_params=None, clip_params=None, beta_params=None, jit_compile=True):
        if not hasattr(self, 'ema_noise'):
            self.ema_noise = None
        
        ema_noise = self.estimate_gradient_variance(self.batch, num_samples, self.ema_noise, smooth, jit_compile)
        
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
    
    
    def adjust(self, target_ess=None, target_noise=None, num_samples=None, smooth=0.2, batch_params=None, alpha_params=None, eps_params=None, tau_params=None, gamma_params=None, store_params=None, clip_params=None, beta_params=None, jit_compile=True):
        if target_noise is None:
            self.adjust_batch_size(smooth, batch_params, target_ess, alpha_params, eps_params, tau_params, gamma_params, store_params, clip_params, beta_params)
        else:
            self.adabatch(num_samples, target_noise, smooth, batch_params, alpha_params, eps_params, tau_params, gamma_params, clip_params, beta_params, jit_compile)
    
    
    def data_func(self, state_pool, action_pool, next_state_pool, reward_pool, done_pool):
        if self.parallel_store_and_training:
            curr_len = self.pool_lengths[p]
            TD_list=[]
            ratio_list=[]
            self.length_list=[]
            for p in range(self.processes):
                TD_list.append(self._get_buffer(p, 'TD')[:self.pool_lengths[p]])
                self.length_list.append(len(TD_list[p]))
                if self.PPO:
                    ratio_list.append(self._get_buffer(p, 'ratio')[:self.length_list[p]])
            TD=np.concat(TD_list, axis=0)
            np.frombuffer(self.shared_TD.get_obj(), dtype=np.float32)[:len(TD)]=TD
            if self.PPO:
                ratio=np.concat(ratio_list, axis=0)
                np.frombuffer(self.shared_ratio.get_obj(), dtype=np.float32)[:len(ratio)]=ratio
            TD_length=len(TD)
            length=min(curr_len,TD_length)
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
                s,a,next_s,r,d=self.prioritized_replay.sample(state_pool,action_pool,next_state_pool,reward_pool,done_pool,self.lambda_,self.alpha,self.batch)
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
    
    
    def dataset_fn(self, dataset, global_batch_size, input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
        dataset = dataset.batch(batch_size)
        return dataset
    
    
    @tf.function(jit_compile=True)
    def train_step(self, train_data, train_loss, optimizer):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.__call__(*train_data)
        if type(optimizer)!=list:
            gradients = tape.gradient(loss, self.param)
            optimizer.apply_gradients(zip(gradients, self.param))
        else:
            for i in range(len(optimizer)):
                gradients = tape.gradient(loss, self.param[i])
                optimizer[i].apply_gradients(zip(gradients, self.param[i]))
        train_loss(loss)
        return loss
      
      
    @tf.function
    def train_step_(self, train_data, train_loss, optimizer):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.__call__(*train_data)
        if type(optimizer)!=list:
            gradients = tape.gradient(loss, self.param)
            optimizer.apply_gradients(zip(gradients, self.param))
        else:
            for i in range(len(optimizer)):
                gradients = tape.gradient(loss, self.param[i])
                optimizer[i].apply_gradients(zip(gradients, self.param[i]))
        train_loss(loss)
        return loss
    
    
    def _train_step(self, train_data, optimizer):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.__call__(*train_data)
            loss = self.compute_loss(loss)
        if type(optimizer)!=list:
            gradients = tape.gradient(loss, self.param)
            optimizer.apply_gradients(zip(gradients, self.param))
        else:
            for i in range(len(optimizer)):
                gradients = tape.gradient(loss, self.param[i])
                optimizer[i].apply_gradients(zip(gradients, self.param[i]))
        return loss 
    
    
    @tf.function(jit_compile=True)
    def distributed_train_step(self, dataset_inputs, optimizer, strategy):
        per_replica_losses = strategy.run(self._train_step, args=(dataset_inputs, optimizer))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)
    
    
    @tf.function
    def distributed_train_step_(self, dataset_inputs, optimizer, strategy):
        per_replica_losses = strategy.run(self._train_step, args=(dataset_inputs, optimizer))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)
    
    
    def CTL(self, multi_worker_dataset, num_steps_per_episode=None):
        iterator = iter(multi_worker_dataset)
        total_loss = 0.0
        num_batches = 0
        if self.PR==True or self.HER==True or self.TRL==True:
            if self.jit_compile==True:
                total_loss = self.distributed_train_step(next(iterator), self.optimizer)
            else:
                total_loss = self.distributed_train_step_(next(iterator), self.optimizer)
            self.prioritized_replay.update()
            self.batch_counter += 1
            if self.pool_network==True:
                if self.batch_counter%self.update_batches==0:
                    self.update_param()
                    if not hasattr(self,'window_size_func'):
                        if self.PPO:
                            window_size=self.window_size_ppo
                        else:
                            window_size=self.window_size_pr
                    if hasattr(self, 'adjust_func'):
                        self._ess = self.compute_ess(None, None)
                    for p in range(self.processes):
                        if self.parallel_store_and_training:
                            self.lock_list[p].acquire()
                        curr_len = self.pool_lengths[p]
                        if hasattr(self,'window_size_func'):
                            window_size=int(self.window_size_func(p))
                            if self.PPO:
                                scores = self.lambda_ * self._get_buffer(p, 'TD')[:curr_len] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:curr_len] - 1.0)
                                weights = scores + 1e-7
                            else:
                                weights = self._get_buffer(p, 'TD')[:curr_len] + 1e-7
                            p=weights/tf.reduce_sum(weights)
                            idx=np.random.choice(np.arange(len(self._get_buffer(p, 'done'))),size=[len(self._get_buffer(p, 'done')[:curr_len])-window_size],p=p.numpy(),replace=False)
                        if window_size!=None and len(self._get_buffer(p, 'done'))>window_size:
                            self._get_buffer(p, 'state')[:len(idx)]=self._get_buffer(p, 'state')[idx]
                            self._get_buffer(p, 'action')[:len(idx)]=self._get_buffer(p, 'action')[idx]
                            self._get_buffer(p, 'next_state')[:len(idx)]=self._get_buffer(p, 'next_state')[idx]
                            self._get_buffer(p, 'reward')[:len(idx)]=self._get_buffer(p, 'reward')[idx]
                            self._get_buffer(p, 'done')[:len(idx)]=self._get_buffer(p, 'done')[idx]
                            self.write_indices[p] = len(idx)
                            self.pool_lengths[p] = len(idx)
                            if self.PPO:
                                self._get_buffer(p, 'ratio')[:len(idx)]=self._get_buffer(p, 'ratio')[idx]
                            self._get_buffer(p, 'TD')[:len(idx)]=self._get_buffer(p, 'TD')[idx]
                            if self.PPO:
                                scores = self.lambda_ * self._get_buffer(p, 'TD')[:len(idx)] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:len(idx)] - 1.0)
                                weights = scores + 1e-7
                            else:
                                weights = self._get_buffer(p, 'TD')[:len(idx)] + 1e-7
                            self.ess_[p] = self.compute_ess_from_weights(weights)
                        if self.parallel_store_and_training:
                            self.lock_list[p].release()
                    curr_len = self.pool_lengths[p]
                    if self.PPO:
                        ratio_list = [self._get_buffer(p, 'ratio')[:curr_len] for p in range(self.processes)]
                        self.prioritized_replay.ratio = np.concat(ratio_list, axis=0)
                        TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                        self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                    else:
                        TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                        self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                        if not self.parallel_store_and_training:
                            state_pools = [self._get_buffer(p, 'state')[:curr_len] for p in range(self.processes)]
                            self.state_pool = np.concatenate(state_pools, axis=0)
                            action_pools = [self._get_buffer(p, 'action')[:curr_len] for p in range(self.processes)]
                            self.action_pool = np.concatenate(action_pools, axis=0)
                            next_state_pools = [self._get_buffer(p, 'next_state')[:curr_len] for p in range(self.processes)]
                            self.next_state_pool = np.concatenate(next_state_pools, axis=0)
                            reward_pools = [self._get_buffer(p, 'reward')[:curr_len] for p in range(self.processes)]
                            self.reward_pool = np.concatenate(reward_pools, axis=0)
                            done_pools = [self._get_buffer(p, 'done')[:curr_len] for p in range(self.processes)]
                            self.done_pool = np.concatenate(done_pools, axis=0)
                if hasattr(self, 'adjust_func') and len(len(self.prioritized_replay.TD))>=self.pool_size_:
                    self.adjust_func()
            return total_loss
        else:
            batch = 0
            while self.step_in_episode < num_steps_per_episode:
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    break
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_begin'):
                        callback.on_batch_begin(batch, logs={})
                if self.jit_compile==True:
                    loss = self.distributed_train_step(next(iterator), self.optimizer)
                else:
                    loss = self.distributed_train_step_(next(iterator), self.optimizer)
                total_loss += loss
                batch_logs = {'loss': loss.numpy()}
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_end'):
                        callback.on_batch_end(batch, logs=batch_logs)
                num_batches += 1
                self.batch_counter += 1
                self.step_in_episode += 1
                batch += 1
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
                    if hasattr(self, 'adjust_func'):
                        self.adjust_func()
                    if self.PPO and self.batch_counter%self.update_batches==0:
                        break
                if self.stop_training==True:
                    return total_loss,num_batches
                if self.save_freq_!=None and self.batch_counter%self.save_freq_==0:
                    self.save_checkpoint()
            return total_loss,num_batches
        
        
    @tf.function(jit_compile=True)
    def per_worker_dataset_fn(self):
        return self.strategy.distribute_datasets_from_function(self.dataset_fn)
    
    
    @tf.function
    def per_worker_dataset_fn_(self):
        return self.strategy.distribute_datasets_from_function(self.dataset_fn)

    
    def CTL_param(self, coordinator, num_steps_per_episode=None):
        if self.jit_compile==True:
            per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_dataset_fn)
        else:
            per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_dataset_fn_)
        per_worker_iterator = iter(per_worker_dataset)
        total_loss = 0.0
        num_batches = 0
        if self.PR==True or self.HER==True or self.TRL==True:
            if self.jit_compile==True:
                total_loss = coordinator.schedule(self.distributed_train_step, args=(next(per_worker_iterator), self.optimizer))
            else:
                total_loss = coordinator.schedule(self.distributed_train_step_, args=(next(per_worker_iterator), self.optimizer))
            self.prioritized_replay.update()
            self.batch_counter += 1
            if self.pool_network==True:
                if self.batch_counter%self.update_batches==0:
                    self.update_param()
                    if not hasattr(self,'window_size_fn'):
                        if self.PPO:
                            window_size=self.window_size_ppo
                        else:
                            window_size=self.window_size_pr
                    if hasattr(self, 'adjust_func'):
                        self._ess = self.compute_ess(None, None)
                    for p in range(self.processes):
                        if self.parallel_store_and_training:
                            self.lock_list[p].acquire()
                        curr_len = self.pool_lengths[p]
                        if hasattr(self,'window_size_func'):
                            window_size=int(self.window_size_func(p))
                            if self.PPO:
                                scores = self.lambda_ * self._get_buffer(p, 'TD')[:curr_len] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:curr_len] - 1.0)
                                weights = scores + 1e-7
                            else:
                                weights = self._get_buffer(p, 'TD')[:curr_len] + 1e-7
                            p=weights/tf.reduce_sum(weights)
                            idx=np.random.choice(np.arange(len(self._get_buffer(p, 'done'))),size=[len(self._get_buffer(p, 'done')[:curr_len])-window_size],p=p.numpy(),replace=False)
                        if window_size!=None and len(self._get_buffer(p, 'done'))>window_size:
                            self._get_buffer(p, 'state')[:len(idx)]=self._get_buffer(p, 'state')[idx]
                            self._get_buffer(p, 'action')[:len(idx)]=self._get_buffer(p, 'action')[idx]
                            self._get_buffer(p, 'next_state')[:len(idx)]=self._get_buffer(p, 'next_state')[idx]
                            self._get_buffer(p, 'reward')[:len(idx)]=self._get_buffer(p, 'reward')[idx]
                            self._get_buffer(p, 'done')[:len(idx)]=self._get_buffer(p, 'done')[idx]
                            self.write_indices[p] = len(idx)
                            self.pool_lengths[p] = len(idx)
                            if self.PPO:
                                self._get_buffer(p, 'ratio')[:len(idx)]=self._get_buffer(p, 'ratio')[idx]
                            self._get_buffer(p, 'TD')[:len(idx)]=self._get_buffer(p, 'TD')[idx]
                            if self.PPO:
                                scores = self.lambda_ * self._get_buffer(p, 'TD')[:len(idx)] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:len(idx)] - 1.0)
                                weights = scores + 1e-7
                            else:
                                weights = self._get_buffer(p, 'TD')[:len(idx)] + 1e-7
                            self.ess_[p] = self.compute_ess_from_weights(weights)
                        if self.parallel_store_and_training:
                            self.lock_list[p].release()
                    curr_len = self.pool_lengths[p]
                    if self.PPO:
                        ratio_list = [self._get_buffer(p, 'ratio')[:curr_len] for p in range(self.processes)]
                        self.prioritized_replay.ratio = np.concat(ratio_list, axis=0)
                        TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                        self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                    else:
                        TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                        self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                        if not self.parallel_store_and_training:
                            state_pools = [self._get_buffer(p, 'state')[:curr_len] for p in range(self.processes)]
                            self.state_pool = np.concatenate(state_pools, axis=0)
                            action_pools = [self._get_buffer(p, 'action')[:curr_len] for p in range(self.processes)]
                            self.action_pool = np.concatenate(action_pools, axis=0)
                            next_state_pools = [self._get_buffer(p, 'next_state')[:curr_len] for p in range(self.processes)]
                            self.next_state_pool = np.concatenate(next_state_pools, axis=0)
                            reward_pools = [self._get_buffer(p, 'reward')[:curr_len] for p in range(self.processes)]
                            self.reward_pool = np.concatenate(reward_pools, axis=0)
                            done_pools = [self._get_buffer(p, 'done')[:curr_len] for p in range(self.processes)]
                            self.done_pool = np.concatenate(done_pools, axis=0)
                if hasattr(self, 'adjust_func') and len(len(self.prioritized_replay.TD))>=self.pool_size_:
                    self.adjust_func()
            return total_loss
        else:
            batch = 0
            while self.step_in_episode < num_steps_per_episode:
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_begin'):
                        callback.on_batch_begin(batch, logs={})
                if self.jit_compile==True:
                    loss = coordinator.schedule(self.distributed_train_step, args=(next(per_worker_iterator), self.optimizer))
                else:
                    loss = coordinator.schedule(self.distributed_train_step_, args=(next(per_worker_iterator), self.optimizer))
                total_loss += loss
                batch_logs = {'loss': loss.fetch()}
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_end'):
                        callback.on_batch_end(batch, logs=batch_logs)
                num_batches += 1
                self.batch_counter += 1
                self.step_in_episode += 1
                batch += 1
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
                    if hasattr(self, 'adjust_func'):
                        self.adjust_func()
                    if self.PPO and self.batch_counter%self.update_batches==0:
                        break
                if self.stop_training==True:
                    coordinator.join()
                    return total_loss,num_batches
                if self.save_freq_!=None and self.batch_counter%self.save_freq_==0:
                    self.save_checkpoint()
            coordinator.join()
            return total_loss,num_batches
    
    
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
            if curr_len>math.ceil(self.pool_size/self.processes):
                self.lock_list[p].acquire()
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
                        self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[window_size:]
                        if self.PPO:
                            self._get_buffer(p, 'ratio')=self._get_buffer(p, 'ratio')[window_size:]
                else:
                    window_size = len(self._get_buffer(p, 'state'))-math.ceil(self.pool_size/self.processes)
                    self._get_buffer(p, 'state')[:curr_len-window_size]=self._get_buffer(p, 'state')[window_size:]
                    self._get_buffer(p, 'action')[:curr_len-window_size]=self._get_buffer(p, 'action')[window_size:]
                    self._get_buffer(p, 'next_state')[:curr_len-window_size]=self._get_buffer(p, 'next_state')[window_size:]
                    self._get_buffer(p, 'reward')[:curr_len-window_size]=self._get_buffer(p, 'reward')[window_size:]
                    self._get_buffer(p, 'done')[:curr_len-window_size]=self._get_buffer(p, 'done')[window_size:]
                    self.write_indices[p] = curr_len-window_size
                    self.pool_lengths[p] = curr_len-window_size
                    if self.PR:
                        self._get_buffer(p, 'TD')=self._get_buffer(p, 'TD')[len(self._get_buffer(p, 'TD'))-math.ceil(self.pool_size/self.processes):]
                        if self.PPO:
                            self._get_buffer(p, 'ratio')=self._get_buffer(p, 'ratio')[len(self._get_buffer(p, 'ratio'))-math.ceil(self.pool_size/self.processes):]
                self.lock_list[p].release()
    
    
    def train1_pr(self, batch, total_loss, num_batches):
        if self.parallel_store_and_training:
            state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.update_pool()
        else:
            state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.state_pool, self.action_pool, self.next_state_pool, self.reward_pool, self.done_pool
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_begin'):
                callback.on_batch_begin(batch, logs={})
        state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func(state_pool, action_pool, next_state_pool, reward_pool, done_pool)
        train_ds=tf.data.Dataset.from_tensor_slices((state_batch,action_batch,next_state_batch,reward_batch,done_batch)).batch(self.batch)
        if isinstance(self.strategy,tf.distribute.MirroredStrategy):
            train_ds=self.strategy.experimental_distribute_dataset(train_ds)
            for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                if self.jit_compile==True:
                    loss=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.optimizer,self.strategy)
                else:
                    loss=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.optimizer,self.strategy)
                if self.parallel_store_and_training:
                    np.frombuffer(self.shared_TD.get_obj(), dtype=np.float32)[self.prioritized_replay.index]=tf.abs(self.prioritized_replay.TD_[:self.prioritized_replay.batch])
                    if self.PPO:
                        np.frombuffer(self.shared_ratio.get_obj(), dtype=np.float32)[self.prioritized_replay.index]=self.prioritized_replay.ratio_[:self.prioritized_replay.batch]
                    self.clear_pool()
                else:
                    self.prioritized_replay.update()
                total_loss+=loss
                num_batches += 1
                self.batch_counter+=1
                if self.pool_network==True:
                    if self.batch_counter%self.update_batches==0:
                        self.update_param()
                        if not hasattr(self,'window_size_func'):
                            if self.PPO:
                                window_size=self.window_size_ppo
                            else:
                                window_size=self.window_size_pr
                        if hasattr(self, 'adjust_func'):
                            self._ess = self.compute_ess(None, None)
                        for p in range(self.processes):
                            if self.parallel_store_and_training:
                                self.lock_list[p].acquire()
                            curr_len = self.pool_lengths[p]
                            if hasattr(self,'window_size_func'):
                                window_size=int(self.window_size_func(p))
                                if self.PPO:
                                    scores = self.lambda_ * self._get_buffer(p, 'TD')[:curr_len] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:curr_len] - 1.0)
                                    weights = scores + 1e-7
                                else:
                                    weights = self._get_buffer(p, 'TD')[:curr_len] + 1e-7
                                p=weights/tf.reduce_sum(weights)
                                idx=np.random.choice(np.arange(len(self._get_buffer(p, 'done'))),size=[len(self._get_buffer(p, 'done')[:curr_len])-window_size],p=p.numpy(),replace=False)
                            if window_size!=None and len(self._get_buffer(p, 'done'))>window_size:
                                self._get_buffer(p, 'state')[:len(idx)]=self._get_buffer(p, 'state')[idx]
                                self._get_buffer(p, 'action')[:len(idx)]=self._get_buffer(p, 'action')[idx]
                                self._get_buffer(p, 'next_state')[:len(idx)]=self._get_buffer(p, 'next_state')[idx]
                                self._get_buffer(p, 'reward')[:len(idx)]=self._get_buffer(p, 'reward')[idx]
                                self._get_buffer(p, 'done')[:len(idx)]=self._get_buffer(p, 'done')[idx]
                                self.write_indices[p] = len(idx)
                                self.pool_lengths[p] = len(idx)
                                if self.PPO:
                                    self._get_buffer(p, 'ratio')[:len(idx)]=self._get_buffer(p, 'ratio')[idx]
                                self._get_buffer(p, 'TD')[:len(idx)]=self._get_buffer(p, 'TD')[idx]
                                if self.PPO:
                                    scores = self.lambda_ * self._get_buffer(p, 'TD')[:len(idx)] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:len(idx)] - 1.0)
                                    weights = scores + 1e-7
                                else:
                                    weights = self._get_buffer(p, 'TD')[:len(idx)] + 1e-7
                                self.ess_[p] = self.compute_ess_from_weights(weights)
                            if self.parallel_store_and_training:
                                self.lock_list[p].release()
                        curr_len = self.pool_lengths[p]
                        if self.PPO:
                            ratio_list = [self._get_buffer(p, 'ratio')[:curr_len] for p in range(self.processes)]
                            self.prioritized_replay.ratio = np.concat(ratio_list, axis=0)
                            TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                            self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                        else:
                            TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                            self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                            if not self.parallel_store_and_training:
                                state_pools = [self._get_buffer(p, 'state')[:curr_len] for p in range(self.processes)]
                                self.state_pool = np.concatenate(state_pools, axis=0)
                                action_pools = [self._get_buffer(p, 'action')[:curr_len] for p in range(self.processes)]
                                self.action_pool = np.concatenate(action_pools, axis=0)
                                next_state_pools = [self._get_buffer(p, 'next_state')[:curr_len] for p in range(self.processes)]
                                self.next_state_pool = np.concatenate(next_state_pools, axis=0)
                                reward_pools = [self._get_buffer(p, 'reward')[:curr_len] for p in range(self.processes)]
                                self.reward_pool = np.concatenate(reward_pools, axis=0)
                                done_pools = [self._get_buffer(p, 'done')[:curr_len] for p in range(self.processes)]
                                self.done_pool = np.concatenate(done_pools, axis=0)
                    if hasattr(self, 'adjust_func') and len(len(self.prioritized_replay.TD))>=self.pool_size_:
                        self.adjust_func()
                    if self.PPO and self.batch_counter%self.update_batches==0:
                        return (total_loss / num_batches).numpy()
        elif isinstance(self.strategy,tf.distribute.MultiWorkerMirroredStrategy):
            with self.strategy.scope():
                multi_worker_dataset = self.strategy.distribute_datasets_from_function(
                    lambda input_context: self.dataset_fn(train_ds, self.batch, input_context))  
            loss=self.CTL(multi_worker_dataset)
            total_loss+=loss
            num_batches += 1
            if self.PPO and self.batch_counter%self.update_batches==0:
                return (total_loss / num_batches).numpy()
        elif isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
            self.state_pool_[:self.batch].assign(state_batch)
            self.action_pool_[:self.batch].assign(action_batch)
            self.next_state_pool_[:self.batch].assign(next_state_batch)
            self.reward_pool_[:self.batch].assign(reward_batch)
            self.done_pool_[:self.batch].assign(done_batch)
            self.batch_.assign(self.batch)
            loss=self.CTL_param(self.coordinator)
            total_loss+=loss
            num_batches += 1
            if self.PPO and self.batch_counter%self.update_batches==0:
                self.coordinator.join()
                return total_loss.fetch() / num_batches
        elif self.distributed_flag!=True:
            if self.jit_compile==True:
                loss=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.train_loss,self.optimizer)
            else:
                loss=self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.train_loss,self.optimizer)
            if self.parallel_store_and_training:
                np.frombuffer(self.shared_TD.get_obj(), dtype=np.float32)[self.prioritized_replay.index]=tf.abs(self.prioritized_replay.TD_[:self.prioritized_replay.batch])
                if self.PPO:
                    np.frombuffer(self.shared_ratio.get_obj(), dtype=np.float32)[self.prioritized_replay.index]=self.prioritized_replay.ratio_[:self.prioritized_replay.batch]
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
                    if hasattr(self, 'adjust_func'):
                        self._ess = self.compute_ess(None, None)
                    for p in range(self.processes):
                        if self.parallel_store_and_training:
                            if self.parallel_store_and_training:
                                self.lock_list[p].acquire()
                            curr_len = self.pool_lengths[p]
                            if hasattr(self,'window_size_func'):
                                window_size=int(self.window_size_func(p))
                                if self.PPO:
                                    scores = self.lambda_ * self._get_buffer(p, 'TD')[:curr_len] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:curr_len] - 1.0)
                                    weights = scores + 1e-7
                                else:
                                    weights = self._get_buffer(p, 'TD')[:curr_len] + 1e-7
                                p=weights/tf.reduce_sum(weights)
                                idx=np.random.choice(np.arange(len(self._get_buffer(p, 'done'))),size=[len(self._get_buffer(p, 'done')[:curr_len])-window_size],p=p.numpy(),replace=False)
                            if window_size!=None and len(self._get_buffer(p, 'done'))>window_size:
                                self._get_buffer(p, 'state')[:len(idx)]=self._get_buffer(p, 'state')[idx]
                                self._get_buffer(p, 'action')[:len(idx)]=self._get_buffer(p, 'action')[idx]
                                self._get_buffer(p, 'next_state')[:len(idx)]=self._get_buffer(p, 'next_state')[idx]
                                self._get_buffer(p, 'reward')[:len(idx)]=self._get_buffer(p, 'reward')[idx]
                                self._get_buffer(p, 'done')[:len(idx)]=self._get_buffer(p, 'done')[idx]
                                self.write_indices[p] = len(idx)
                                self.pool_lengths[p] = len(idx)
                                if self.PPO:
                                    self._get_buffer(p, 'ratio')[:len(idx)]=self._get_buffer(p, 'ratio')[idx]
                                self._get_buffer(p, 'TD')[:len(idx)]=self._get_buffer(p, 'TD')[idx]
                                if self.PPO:
                                    scores = self.lambda_ * self._get_buffer(p, 'TD')[:len(idx)] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:len(idx)] - 1.0)
                                    weights = scores + 1e-7
                                else:
                                    weights = self._get_buffer(p, 'TD')[:len(idx)] + 1e-7
                                self.ess_[p] = self.compute_ess_from_weights(weights)
                            if self.parallel_store_and_training:
                                self.lock_list[p].release()
                        curr_len = self.pool_lengths[p]
                        if self.PPO:
                            ratio_list = [self._get_buffer(p, 'ratio')[:curr_len] for p in range(self.processes)]
                            self.prioritized_replay.ratio = np.concat(ratio_list, axis=0)
                            TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                            self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                        else:
                            TD_list = [self._get_buffer(p, 'TD')[:curr_len] for p in range(self.processes)]
                            self.prioritized_replay.TD = np.concat(TD_list, axis=0)
                            if not self.parallel_store_and_training:
                                state_pools = [self._get_buffer(p, 'state')[:curr_len] for p in range(self.processes)]
                                self.state_pool = np.concatenate(state_pools, axis=0)
                                action_pools = [self._get_buffer(p, 'action')[:curr_len] for p in range(self.processes)]
                                self.action_pool = np.concatenate(action_pools, axis=0)
                                next_state_pools = [self._get_buffer(p, 'next_state')[:curr_len] for p in range(self.processes)]
                                self.next_state_pool = np.concatenate(next_state_pools, axis=0)
                                reward_pools = [self._get_buffer(p, 'reward')[:curr_len] for p in range(self.processes)]
                                self.reward_pool = np.concatenate(reward_pools, axis=0)
                                done_pools = [self._get_buffer(p, 'done')[:curr_len] for p in range(self.processes)]
                                self.done_pool = np.concatenate(done_pools, axis=0)
                    if hasattr(self, 'adjust_func') and len(len(self.prioritized_replay.TD))>=self.pool_size_:
                        self.adjust_func()
                if self.PPO and self.batch_counter%self.update_batches==0:
                    return self.train_loss.result().numpy()
        if self.save_freq_!=None and self.batch_counter%self.save_freq_==0:
            self.save_checkpoint()
        if not isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
            batch_logs = {'loss': loss.numpy()}
        else:
            batch_logs = {'loss': loss.fetch()}
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(batch, logs=batch_logs)
        return total_loss, num_batches
    
    
    def train1(self):
        self.step_counter+=1
        self.end_flag=False
        batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
        if len(self.state_pool)%self.batch!=0:
            batches+=1
        if self.PR==True or self.HER==True or self.TRL==True:
            total_loss = 0.0
            num_batches = 0
            batch = 0
            for j in range(batches):
                if self.stop_training==True:
                    if self.distributed_flag==True:
                        if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                            self.coordinator.join()
                            return total_loss.fetch() / num_batches
                        else:
                            return (total_loss / num_batches).numpy()
                    else:
                        return self.train_loss.result().numpy()
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    break
                batch += 1
                total_loss, num_batches = self.train1_pr(batch, total_loss, num_batches)
                if self.PPO and self.batch_counter%self.update_batches==0:
                    return total_loss
            if len(self.state_pool)%self.batch!=0:
                if self.stop_training==True:
                    if self.distributed_flag==True:
                        if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                            self.coordinator.join()
                            return total_loss.fetch() / num_batches
                        else:
                            return (total_loss / num_batches).numpy()
                    else:
                        return self.train_loss.result().numpy()
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    if self.distributed_flag==True:
                        if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                            return total_loss.fetch() / num_batches
                        else:
                            return (total_loss / num_batches).numpy()
                    else:
                        return self.train_loss.result().numpy()
                batch += 1
                total_loss, num_batches = self.train1_pr(batch, total_loss, num_batches)
                if self.PPO and self.batch_counter%self.update_batches==0:
                    return total_loss
            if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                self.coordinator.join()
        else:
            batch = 0
            if self.distributed_flag==True:
                total_loss = 0.0
                num_batches = 0
                if self.parallel_store_and_training:
                    state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.update_pool()
                else:
                    state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.state_pool, self.action_pool, self.next_state_pool, self.reward_pool, self.done_pool
                if self.pool_network==True:
                    if self.parallel_store_and_training:
                        length=len(done_pool)
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool[:length],action_pool[:length],next_state_pool[:length],reward_pool[:length],done_pool[:length])).batch(self.batch)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool,action_pool,next_state_pool,reward_pool,done_pool)).batch(self.batch)
                else:
                    if self.num_updates!=None:
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool,action_pool,next_state_pool,reward_pool,done_pool)).batch(self.batch)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool,action_pool,next_state_pool,reward_pool,done_pool)).shuffle(len(state_pool)).batch(self.batch)
                if isinstance(self.strategy,tf.distribute.MirroredStrategy):
                    train_ds=self.strategy.experimental_distribute_dataset(train_ds)
                    for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                        if self.stop_training==True:
                            return (total_loss / num_batches).numpy()
                        if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                            break
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_batch_begin'):
                                callback.on_batch_begin(batch, logs={})
                        if self.jit_compile==True:
                            loss=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.optimizer,self.strategy)
                        else:
                            loss=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.optimizer,self.strategy)
                        total_loss+=loss
                        batch_logs = {'loss': loss.numpy()}
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_batch_end'):
                                callback.on_batch_end(batch, logs=batch_logs)
                        num_batches += 1
                        self.batch_counter += 1
                        batch += 1
                        if self.pool_network==True:
                            if self.batch_counter%self.update_batches==0:
                                self.update_param()
                                if not hasattr(self,'window_size_func'):
                                    if self.PPO:
                                        window_size=self.window_size_ppo
                                    else:
                                        window_size=self.window_size_pr
                                if hasattr(self, 'adjust_func'):
                                    self._ess = self.compute_ess(None, None)
                                if self.PPO:
                                    for p in range(self.processes):
                                        if self.parallel_store_and_training:
                                            self.lock_list[p].acquire()
                                        self.write_indices[p]=0
                                        self.pool_lengths[p]=0
                                        if self.parallel_store_and_training:
                                            self.lock_list[p].release()
                            if hasattr(self, 'adjust_func'):
                                self.adjust_func()
                            if self.PPO and self.batch_counter%self.update_batches==0:
                                break
                        if self.save_freq_!=None and self.batch_counter%self.save_freq_==0:
                            self.save_checkpoint()
                elif isinstance(self.strategy,tf.distribute.MultiWorkerMirroredStrategy):
                    with self.strategy.scope():
                        multi_worker_dataset = self.strategy.distribute_datasets_from_function(
                            lambda input_context: self.dataset_fn(train_ds, self.batch, input_context))  
                    total_loss,num_batches=self.CTL(multi_worker_dataset,math.ceil(len(self.state_pool)/self.batch))
                    if self.PPO and self.batch_counter%self.update_batches==0:
                        return (total_loss / num_batches).numpy()
                elif isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                    total_loss,num_batches=self.CTL_param(self.coordinator,math.ceil(len(self.state_pool)/self.batch))
                    if self.PPO and self.batch_counter%self.update_batches==0:
                        self.coordinator.join()
                        return total_loss.fetch() / num_batches
            else:
                if self.parallel_store_and_training:
                    state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.update_pool()
                else:
                    state_pool, action_pool, next_state_pool, reward_pool, done_pool = self.state_pool, self.action_pool, self.next_state_pool, self.reward_pool, self.done_pool
                if self.pool_network==True:
                    if self.parallel_store_and_training:
                        length=len(done_pool)
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool[:length],action_pool[:length],next_state_pool[:length],reward_pool[:length],done_pool[:length])).batch(self.batch)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool,action_pool,next_state_pool,reward_pool,done_pool)).batch(self.batch)
                else:
                    if self.num_updates!=None:
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool,action_pool,next_state_pool,reward_pool,done_pool)).batch(self.batch)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool,action_pool,next_state_pool,reward_pool,done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                    if self.stop_training==True:
                        return self.train_loss.result().numpy() 
                    if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_batch_begin'):
                            callback.on_batch_begin(batch, logs={})
                    if self.jit_compile==True:
                        loss=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.train_loss,self.optimizer)
                    else:
                        loss=self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],self.train_loss,self.optimizer)
                    batch_logs = {'loss': loss.numpy()}
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_batch_end'):
                            callback.on_batch_end(batch, logs=batch_logs)
                    num_batches += 1
                    self.batch_counter += 1
                    batch += 1
                    if self.pool_network==True:
                        if self.batch_counter%self.update_batches==0:
                            self.update_param()
                            if not hasattr(self,'window_size_func'):
                                if self.PPO:
                                    window_size=self.window_size_ppo
                                else:
                                    window_size=self.window_size_pr
                            if hasattr(self, 'adjust_func'):
                                self._ess = self.compute_ess(None, None)
                            if self.PPO:
                                for p in range(self.processes):
                                    if self.parallel_store_and_training:
                                        self.lock_list[p].acquire()
                                    self.write_indices[p]=0
                                    self.pool_lengths[p]=0
                                    if self.parallel_store_and_training:
                                        self.lock_list[p].release()
                        if hasattr(self, 'adjust_func'):
                            self.adjust_func()
                        if self.PPO and self.batch_counter%self.update_batches==0:
                            break
                    if self.save_freq_!=None and self.batch_counter%self.save_freq_==0:
                        self.save_checkpoint()
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
                            scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
                            weights = scores + 1e-7
                        else:
                            weights = self.prioritized_replay.TD + 1e-7
                        p=weights/tf.reduce_sum(weights)
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
                if self.distributed_flag==True:
                    if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                        return total_loss.fetch() / num_batches
                    else:
                        return (total_loss / num_batches).numpy()
                else:
                    return self.train_loss.result().numpy() 
        elif self.pool_network==False:
            self.update_param()
        if self.distributed_flag==True:
            if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                return total_loss.fetch() / num_batches
            else:
                return (total_loss / num_batches).numpy()
        else:
            return self.train_loss.result().numpy()
    
    
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
                        scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
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
                state,action,next_state,reward,done=self.prioritized_replay.sample(self.state_pool[7],self.action_pool[7],self.next_state_pool[7],self.reward_pool[7],self.done_pool[7],self.lambda_,self.alpha,int(self.batch/self.processes_pr))
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
                    inverse_len=tf.constant(self.inverse_len)
                    total_inverse=tf.reduce_sum(inverse_len)
                    prob=inverse_len/total_inverse
                    p=np.random.choice(self.processes,p=prob.numpy(),replace=False)
                    self.inverse_len[p]=1/(len(self.state_pool_list[p])+1)
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
                    process=mp.Process(target=self.store_in_parallel,args=(p))
                    process.start()
                    process_list.append(process)
                for process in process_list:
                    process.join()
            counter+=1
            if self.parallel_store_and_training and counter==num_store:
                break
            if not self.parallel_store_and_training:
                if self.state_pool is not None and len(self.state_pool)>=self.batch and counter<num_store:
                        continue
                if self.processes_her==None and self.processes_pr==None:
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
                    if counter<num_store and len(self.state_pool)<self.batch:
                        continue
                    if not isinstance(self.strategy,tf.distribute.ParameterServerStrategy) and not self.PR and self.num_updates!=None:
                        if len(self.state_pool)>=self.pool_size_:
                            idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                        else:
                            idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                        self.state_pool=self.state_pool[idx]
                        self.action_pool=self.action_pool[idx]
                        self.next_state_pool=self.next_state_pool[idx]
                        self.reward_pool=self.reward_pool[idx]
                        self.done_pool=self.done_pool[idx]
                    elif isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                        if self.num_updates!=None:
                            if len(self.state_pool)>=self.pool_size_:
                                idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                            else:
                                idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                            self.state_pool_[:len(idx)].assign(self.state_pool[idx])
                            self.action_pool_[:len(idx)].assign(self.action_pool[idx])
                            self.next_state_pool_[:len(idx)].assign(self.next_state_pool[idx])
                            self.reward_pool_[:len(idx)].assign(self.reward_pool[idx])
                            self.done_pool_[:len(idx)].assign(self.done_pool[idx])
                            self.batch_.assign(self.batch)
                        else:
                            self.state_pool_[:len(self.state_pool)].assign(self.state_pool)
                            self.action_pool_[:len(self.state_pool)].assign(self.action_pool)
                            self.next_state_pool_[:len(self.state_pool)].assign(self.next_state_pool)
                            self.reward_pool_[:len(self.state_pool)].assign(self.reward_pool)
                            self.done_pool_[:len(self.state_pool)].assign(self.done_pool)
                            self.batch_.assign(self.batch)
                else:
                    self.state_pool[7]=np.concatenate(self.state_pool_list)
                    self.action_pool[7]=np.concatenate(self.action_pool_list)
                    self.next_state_pool[7]=np.concatenate(self.next_state_pool_list)
                    self.reward_pool[7]=np.concatenate(self.reward_pool_list)
                    self.done_pool[7]=np.concatenate(self.done_pool_list)
                    if counter<num_store and len(self.state_pool[7])<self.batch:
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
        if not self.parallel_store_and_training:
            if hasattr(self,'window_size_func'):
                for p in range(self.processes):
                    if not hasattr(self,'ess_'):
                        self.ess_ = [None] * self.processes
                    if self.PPO:
                        scores = self.lambda_ * self.TD_list[p] + (1.0-self.lambda_) * tf.abs(self.ratio_list[p] - 1.0)
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
    
    
    def update_pool(self, state_pool, action_pool, next_state_pool, reward_pool, done_pool):
        if not self.end_flag:
            if all(self.end_flag_list):
                self.end_flag=True
                for p in range(self.processes):
                    self.end_flag_list[p]=False
                    self.done_length[p]=0
            if self.processes_her==None and self.processes_pr==None:
                curr_len = self.pool_lengths[p]
                state_pools = [self._get_buffer(p, 'state')[:curr_len] for p in range(self.processes)]
                state_pool = np.concatenate(state_pools, axis=0)
                action_pools = [self._get_buffer(p, 'action')[:curr_len] for p in range(self.processes)]
                action_pool = np.concatenate(action_pools, axis=0)
                next_state_pools = [self._get_buffer(p, 'next_state')[:curr_len] for p in range(self.processes)]
                next_state_pool = np.concatenate(next_state_pools, axis=0)
                reward_pools = [self._get_buffer(p, 'reward')[:curr_len] for p in range(self.processes)]
                reward_pool = np.concatenate(reward_pools, axis=0)
                done_pools = [self._get_buffer(p, 'done')[:curr_len] for p in range(self.processes)]
                done_pool = np.concatenate(done_pools, axis=0)
                if not isinstance(self.strategy,tf.distribute.ParameterServerStrategy) and not self.PR and self.num_updates!=None:
                    if curr_len>=self.pool_size_:
                        idx=np.random.choice(done_pool.shape[0], size=self.pool_size_, replace=False)
                    else:
                        idx=np.random.choice(done_pool.shape[0], size=done_pool.shape[0], replace=False)
                    state_pool=state_pool[idx]
                    action_pool=action_pool[idx]
                    next_state_pool=next_state_pool[idx]
                    reward_pool=reward_pool[idx]
                    done_pool=done_pool[idx]
                elif isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                    if self.num_updates!=None:
                        if curr_len>=self.pool_size_:
                            idx=np.random.choice(done_pool.shape[0], size=self.pool_size_, replace=False)
                        else:
                            idx=np.random.choice(done_pool.shape[0], size=done_pool.shape[0], replace=False)
                        self.state_pool_[:len(idx)].assign(state_pool[idx])
                        self.action_pool_[:len(idx)].assign(action_pool[idx])
                        self.next_state_pool_[:len(idx)].assign(next_state_pool[idx])
                        self.reward_pool_[:len(idx)].assign(reward_pool[idx])
                        self.done_pool_[:len(idx)].assign(done_pool[idx])
                        self.batch_.assign(self.batch)
                    else:
                        self.state_pool_[:len(done_pool)].assign(state_pool)
                        self.action_pool_[:len(done_pool)].assign(action_pool)
                        self.next_state_pool_[:len(done_pool)].assign(next_state_pool)
                        self.reward_pool_[:len(done_pool)].assign(reward_pool)
                        self.done_pool_[:len(done_pool)].assign(done_pool)
                        self.batch_.assign(self.batch)
            else:
                self.state_pool[7]=np.concatenate(self.state_pool_list)
                self.action_pool[7]=np.concatenate(self.action_pool_list)
                self.next_state_pool[7]=np.concatenate(self.next_state_pool_list)
                self.reward_pool[7]=np.concatenate(self.reward_pool_list)
                self.done_pool[7]=np.concatenate(self.done_pool_list)
                if not self.PR and self.num_updates!=None:
                    if curr_len>=self.pool_size_:
                        idx=np.random.choice(self.state_pool[7].shape[0], size=self.pool_size_, replace=False)
                    else:
                        idx=np.random.choice(self.state_pool[7].shape[0], size=self.state_pool[7].shape[0], replace=False)
                    self.state_pool[7]=self.state_pool[7][idx]
                    self.action_pool[7]=self.action_pool[7][idx]
                    self.next_state_pool[7]=self.next_state_pool[7][idx]
                    self.reward_pool[7]=self.reward_pool[7][idx]
                    self.done_pool[7]=self.done_pool[7][idx]
        if self.end_flag:
            if hasattr(self,'window_size_func'):
                for p in range(self.processes):
                    if not hasattr(self,'ess_'):
                        self.ess_ = [None] * self.processes
                    if self.PPO:
                        scores = self.lambda_ * self._get_buffer(p, 'TD')[:curr_len] + (1.0-self.lambda_) * tf.abs(self._get_buffer(p, 'ratio')[:curr_len] - 1.0)
                        weights = scores + 1e-7
                    else:
                        weights = self._get_buffer(p, 'TD')[:curr_len] + 1e-7
                    self.ess_[p] = self.compute_ess_from_weights(weights)
            self.initialize_adjusting()
            if self.PR==True:
                if hasattr(self, 'adjust_func') and curr_len>=self.pool_size_:
                    self.ess.value=self.compute_ess(None,None)
        if self.processes_her==None and self.processes_pr==None:
            if not isinstance(self.strategy,tf.distribute.ParameterServerStrategy) and not self.PR and self.num_updates!=None:
                return state_pool, action_pool, next_state_pool, reward_pool, done_pool
        else:
            return None, None, None, None, None
    
    
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
    
    
    def train(self, train_loss, optimizer=None, episodes=None, pool_network=True, parallel_store_and_training=True, parallel_training_and_save=False, parallel_dump=False, processes=None, num_store=1, processes_her=None, processes_pr=None, window_size=None, clearing_freq=None, window_size_=None, window_size_ppo=None, window_size_pr=None, jit_compile=True, random=False, save_data=True, callbacks=None, p=None):
        self.avg_reward=None
        if p!=0:
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
        self.train_loss=train_loss
        if optimizer!=None:
            self.optimizer=optimizer
        self.episodes=episodes
        self.pool_network=pool_network
        self.parallel_store_and_training=parallel_store_and_training
        self.parallel_training_and_save=parallel_training_and_save
        self.parallel_dump=parallel_dump
        if parallel_dump:
            manager=mp.Manager()
            self.lock=mp.Lock()
            self.param_index_list=manager.list()
            self.state_index_list=manager.list()
        if parallel_store_and_training:
            if self.PR and self.PPO:
                self.share_TD=mp.Array('f', self.pool_size)
                self.share_ratio=mp.Array('f', self.pool_size)
            elif self.PR:
                self.share_TD=mp.Array('f', self.pool_size)
            self.done_length=manager.list([0 for _ in range(processes)])
            self.ess=mp.Value('f',0)
            self.original_num_store=self.num_store
            self.num_store=mp.Value('i',self.num_store)
            self.ess_=manager.list([None for _ in range(processes)])
            self.end_flag_list=manager.list([False for _ in range(processes)])
        if parallel_training_and_save:
            manager=mp.Manager()
            self.param_save_flag_list=manager.list()
            self.state_save_flag_list=manager.list()
            self.save_flag=mp.Value('b',False)
            self.path_list_=manager.list()
        self.processes=processes
        self.num_store=num_store
        self.processes_her=processes_her
        self.processes_pr=processes_pr
        self.window_size=window_size
        self.clearing_freq=clearing_freq
        self.window_size_=window_size_
        self.window_size_ppo=window_size_ppo
        self.window_size_pr=window_size_pr
        self.jit_compile=jit_compile
        self.random=random
        if self.num_updates!=None:
            self.pool_size_=self.num_updates*self.batch
        self.save_data=save_data
        self.p=p
        self.info_flag=0
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
                if self.clearing_freq!=None:
                    self.store_counter=manager.list(self.store_counter)
            else:
                self.pool_lengths = manager.list([0 for _ in range(processes)])
                self.write_indices = manager.list([0 for _ in range(processes)])
                self.inverse_len=manager.list([0 for _ in range(processes)])
                if self.clearing_freq!=None:
                    self.store_counter=manager.list()
            if not save_data:
                for _ in range(processes):
                    if self.clearing_freq!=None:
                        self.store_counter.append(0)
            self.reward=manager.list([0 for _ in range(processes)])
            if parallel_store_and_training or self.HER!=True or self.TRL!=True:
                self.lock_list=manager.list([manager.Lock() for _ in range(processes)])
            if self.PR==True:
                if self.PPO:
                    self.prioritized_replay.ratio=None
                    self.prioritized_replay.TD=None
                else:
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
        self.distributed_flag=False
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs={})
        if episodes!=None:
            for i in range(episodes):
                t1=time.time()
                self.check_early_stopping()
                if self.stop_training==True:
                    break
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_begin'):
                        callback.on_episode_begin(i, logs={})
                train_loss.reset_states()
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
                episode_logs = {'loss': loss}
                episode_logs['reward'] = self.reward_list[-1]
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_end'):
                        callback.on_episode_end(i, logs=episode_logs)
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if self.save_freq_==None and i%self.save_freq==0:
                    self.save_checkpoint()
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
                            if p!=0:
                                print('episode:{0}'.format(self.total_episode))
                                print('average reward:{0}'.format(self.avg_reward))
                                print()
                                print('time:{0}s'.format(self.total_time))
                            return
                if p!=0:
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
                self.check_early_stopping()
                if self.stop_training==True:
                    break
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_begin'):
                        callback.on_episode_begin(i, logs={})
                train_loss.reset_states()
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
                episode_logs = {'loss': loss}
                episode_logs['reward'] = self.reward_list[-1]
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_end'):
                        callback.on_episode_end(i, logs=episode_logs)
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if self.save_freq_==None and i%self.save_freq==0:
                    self.save_checkpoint()
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
                            if p!=0:
                                print('episode:{0}'.format(self.total_episode))
                                print('average reward:{0}'.format(self.avg_reward))
                                print()
                                print('time:{0}s'.format(self.total_time))
                            return
                if p!=0:
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
        if parallel_training_and_save:
            t1=time.time()
            while True:
                if self.save_param_only==False:
                    self.save_flag.value=all(self.param_save_flag_list) and all(self.state_save_flag_list)
                else:
                    self.save_flag.value=all(self.param_save_flag_list)
                if self.save_flag.value:
                    t2=time.time()
                    self.time+=(t2-t1)
                    self._time=self.time-int(self.time)
                    if self._time<0.5:
                        self.time=int(self.time)
                    else:
                        self.time=int(self.time)+1
                    self.total_time+=self.time
                    if p!=0:
                        print('time:{0}s'.format(self.time))
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_train_end'):
                            callback.on_train_end(logs={})
                    return
        else:
            self._time=self.time-int(self.time)
            if self._time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            self.total_time+=self.time
            if p!=0:
                print('time:{0}s'.format(self.time))
            for callback in self.callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end(logs={})
        return
    
    
    def distributed_training(self, optimizer=None, strategy=None, episodes=None, num_episodes=None, pool_network=True, parallel_store_and_training=True, parallel_training_and_save=False, parallel_dump=False, processes=None, num_store=1, processes_her=None, processes_pr=None, window_size=None, clearing_freq=None, window_size_=None, window_size_ppo=None, window_size_pr=None, jit_compile=True, random=False, save_data=True, callbacks=None, p=None):
        self.avg_reward=None
        if num_episodes!=None:
            episodes=num_episodes
        if p!=0:
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
        if optimizer!=None:
            self.optimizer=optimizer
        self.strategy=strategy
        self.episodes=episodes
        self.num_episodes=num_episodes
        self.pool_network=pool_network
        if pool_network:
            manager=mp.Manager()
        self.parallel_store_and_training=parallel_store_and_training
        self.parallel_training_and_save=parallel_training_and_save
        self.parallel_pickle=parallel_dump
        if parallel_dump:
            manager=mp.Manager()
            self.lock=mp.Lock()
            self.param_index_list=manager.list()
            self.state_index_list=manager.list()
        if parallel_store_and_training:
            if self.PR and self.PPO:
                self.share_TD=mp.Array('f', self.pool_size)
                self.share_ratio=mp.Array('f', self.pool_size)
            elif self.PR:
                self.share_TD=mp.Array('f', self.pool_size)
            self.done_length=manager.list([0 for _ in range(processes)])
            self.ess=mp.Value('f',0)
            self.original_num_store=self.num_store
            self.num_store=mp.Value('i',self.num_store)
            self.ess_=manager.list([None for _ in range(processes)])
            self.end_flag_list=manager.list([False for _ in range(processes)])
        if parallel_training_and_save:
            manager=mp.Manager()
            self.param_save_flag_list=mp.list()
            self.state_save_flag_list=mp.list()
            self.save_flag=mp.Value('b',False)
            self.path_list_=manager.list()
        self.processes=processes
        self.num_store=num_store
        self.processes_her=processes_her
        self.processes_pr=processes_pr
        self.window_size=window_size
        self.clearing_freq=clearing_freq
        self.window_size_=window_size_
        self.window_size_ppo=window_size_ppo
        self.window_size_pr=window_size_pr
        self.jit_compile=jit_compile
        self.random=random
        if self.num_updates!=None:
            self.pool_size_=self.num_updates*self.batch
        self.save_data=save_data
        self.p=p
        self.info_flag=1
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
            self.pool_lengths = manager.list([0 for _ in range(processes)])
            self.write_indices = manager.list([0 for _ in range(processes)])
            self._init_shared_experience_buffers(processes)
            if save_data and len(self.shared_states)!=0:
                for p in range(processes):
                    self.shared_states[p] = np.frombuffer(self.shared_states[p].get_obj(), dtype=np.float32)
                    self.shared_next_states[p] = np.frombuffer(self.shared_next_states[p].get_obj(), dtype=np.float32)
                    self.shared_actions[p] = np.frombuffer(self.shared_actions[p].get_obj(), dtype=np.float32)
                    self.shared_rewards[p] = np.frombuffer(self.shared_rewards[p].get_obj(), dtype=np.float32)
                    self.shared_dones[p] = np.frombuffer(self.shared_dones[p].get_obj(), dtype=np.float32)
                if self.clearing_freq!=None:
                    self.store_counter=manager.list(self.store_counter)
            else:
                self.inverse_len=manager.list([0 for _ in range(processes)])
                if self.clearing_freq!=None:
                    self.store_counter=manager.list()
            if not save_data or len(self.state_pool_list)==0:
                for _ in range(processes):
                    if self.clearing_freq!=None:
                        self.store_counter.append(0)
            self.reward=manager.list([0 for _ in range(processes)])
            if parallel_store_and_training or self.HER!=True or self.TRL!=True:
                self.lock_list=[manager.Lock() for _ in range(processes)]
            if self.PR==True:
                if self.PPO:
                    self.prioritized_replay.ratio=None
                    self.prioritized_replay.TD=None
                else:
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
        self.distributed_flag=True
        with strategy.scope():
            def compute_loss(self, per_example_loss):
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch)
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs={})
        if isinstance(strategy,tf.distribute.MirroredStrategy):
            if episodes!=None:
                for i in range(episodes):
                    t1=time.time()
                    self.check_early_stopping()
                    if self.stop_training==True:
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
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
                    episode_logs = {'loss': loss}
                    episode_logs['reward'] = self.reward_list[-1]
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_end'):
                            callback.on_episode_end(i, logs=episode_logs)
                    self.loss=loss
                    self.loss_list.append(loss)
                    self.total_episode+=1
                    if self.save_freq_==None and i%self.save_freq==0:
                        self.save_checkpoint()
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
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(self.avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
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
                    self.check_early_stopping()
                    if self.stop_training==True:
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
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
                    episode_logs = {'loss': loss}
                    episode_logs['reward'] = self.reward_list[-1]
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_end'):
                            callback.on_episode_end(i, logs=episode_logs)
                    self.loss=loss
                    self.loss_list.append(loss)
                    i+=1
                    self.total_episode+=1
                    if self.save_freq_==None and i%self.save_freq==0:
                        self.save_checkpoint()
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
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(self.avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
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
        elif isinstance(strategy,tf.distribute.MultiWorkerMirroredStrategy):
            if num_episodes!=None:
                episode = 0
                self.step_in_episode = 0
                while episode < num_episodes:
                    t1=time.time()
                    self.check_early_stopping()
                    if self.stop_training==True:
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
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
                        
                    if self.save_freq_==None and episode%self.save_freq==0:
                        self.save_checkpoint()
                  
                    episode += 1
                    self.step_in_episode = 0
                    
                    episode_logs = {'loss': loss}
                    episode_logs['reward'] = self.reward_list[-1]
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_end'):
                            callback.on_episode_end(i, logs=episode_logs)
                    self.loss=loss
                    self.loss_list.append(loss)
                    self.total_episode+=1
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
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(self.avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
                        if episode%p==0:
                            if len(self.state_pool)>=self.batch:
                                print('episode:{0}   loss:{1:.4f}'.format(episode+1,loss))
                            if self.avg_reward!=None:
                                print('episode:{0}   average reward:{1}'.format(episode+1,self.avg_reward))
                            else:
                                print('episode:{0}   reward:{1}'.format(episode+1,self.reward))
                            print()
                    t2=time.time()
                    self.time+=(t2-t1)
        elif isinstance(strategy,tf.distribute.ParameterServerStrategy):
            self.coordinator=tf.distribute.coordinator.ClusterCoordinator(strategy)
            if num_episodes!=None:
                episode = 0
                self.step_in_episode = 0
                while episode < num_episodes:
                    t1=time.time()
                    self.check_early_stopping()
                    if self.stop_training==True:
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
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
                        
                    if self.save_freq_==None and episode%self.save_freq==0:
                        self.save_checkpoint()
                  
                    episode += 1
                    self.step_in_episode = 0
                    
                    episode_logs = {'loss': loss}
                    episode_logs['reward'] = self.reward_list[-1]
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_end'):
                            callback.on_episode_end(i, logs=episode_logs)
                    self.loss=loss
                    self.loss_list.append(loss)
                    self.total_episode+=1
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
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(self.avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
                        if episode%p==0:
                            if len(self.state_pool)>=self.batch:
                                print('episode:{0}   loss:{1:.4f}'.format(episode+1,loss))
                            if self.avg_reward!=None:
                                print('episode:{0}   average reward:{1}'.format(episode+1,self.avg_reward))
                            else:
                                print('episode:{0}   reward:{1}'.format(episode+1,self.reward))
                            print()
                    t2=time.time()
                    self.time+=(t2-t1)
            else:
                episode = 0
                self.step_in_episode = 0
                while True:
                    t1=time.time()
                    self.check_early_stopping()
                    if self.stop_training==True:
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
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
                        
                    if self.save_freq_==None and episode%self.save_freq==0:
                        self.save_checkpoint()
                  
                    episode += 1
                    self.step_in_episode = 0
                    
                    episode_logs = {'loss': loss}
                    episode_logs['reward'] = self.reward_list[-1]
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_end'):
                            callback.on_episode_end(i, logs=episode_logs)
                    self.loss=loss
                    self.loss_list.append(loss)
                    self.total_episode+=1
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
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(self.avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
                        if episode%p==0:
                            if len(self.state_pool)>=self.batch:
                                print('episode:{0}   loss:{1:.4f}'.format(episode+1,loss))
                            if self.avg_reward!=None:
                                print('episode:{0}   average reward:{1}'.format(episode+1,self.avg_reward))
                            else:
                                print('episode:{0}   reward:{1}'.format(episode+1,self.reward))
                            print()
                    t2=time.time()
                    self.time+=(t2-t1)
        if parallel_training_and_save:
            t1=time.time()
            while True:
                if self.save_param_only==False:
                    self.save_flag.value=all(self.param_save_flag_list) and all(self.state_save_flag_list)
                else:
                    self.save_flag.value=all(self.param_save_flag_list)
                if self.save_flag.value:
                    t2=time.time()
                    self.time+=(t2-t1)
                    self._time=self.time-int(self.time)
                    if self._time<0.5:
                        self.time=int(self.time)
                    else:
                        self.time=int(self.time)+1
                    self.total_time+=self.time
                    if p!=0:
                        print('time:{0}s'.format(self.time))
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_train_end'):
                            callback.on_train_end(logs={})
                    return
        else:
            self._time=self.time-int(self.time)
            if self._time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            self.total_time+=self.time
            if p!=0:
                print('time:{0}s'.format(self.time))
            for callback in self.callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end(logs={})
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
    
    
    def summary(self):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        total_memory = 0  # Memory usage in bytes
        
        param_flat=nest.flatten(self.param)
        for param in param_flat:
            param_count = tf.size(param).numpy()
            param_dtype_size = param.dtype.size
            param_memory = param_count * param_dtype_size

            total_params += param_count
            total_memory += param_memory

            if param.trainable:
                trainable_params += param_count
            else:
                non_trainable_params += param_count

        def format_memory(bytes_size):
            units = ['Bytes', 'KB', 'MB', 'GB']
            index = 0
            while bytes_size >= 1024 and index < len(units) - 1:
                bytes_size /= 1024
                index += 1
            return f"{bytes_size:.2f} {units[index]}"

        # Print the summary with formatted memory usage
        print("Model Summary")
        print("-------------")
        print(f"Total params: {total_params} ({format_memory(total_memory)})")
        print(f"Trainable params: {trainable_params} ({format_memory(trainable_params * param_dtype_size)})")
        print(f"Non-trainable params: {non_trainable_params} ({format_memory(non_trainable_params * param_dtype_size)})")
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
        if self.save_top_k is None:
            if self.max_save_files==1:
                path=path
            else:
                if self.avg_reward!=None:
                    path=path.replace(path[path.find('.'):],'-{0:.4f}.dat'.format(self.avg_reward))
            self.path_list.append(path)
            if len(self.path_list)>self.max_save_files:
                os.remove(self.path_list[0])
                del self.path_list[0]
            self.save_param(path)
        else:
            if len(self.reward_list)>=self.trial_count:
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                if self.best_avg_reward==None:
                    self.best_avg_reward=avg_reward
                elif avg_reward>self.best_avg_reward:
                    self.path_list.append(path)
                    if len(self.path_list)>self.save_top_k:
                        os.remove(self.path_list[0])
                        del self.path_list[0]
                    self.best_avg_reward=avg_reward
                    self.save_param(path)
        if self.save_last:
            path=self.path+'-last.dat'
            self.save_param(path)
        return
    
    
    def save_param(self,path):
        if self.parallel_training_and_save:
            self.save_flag.value=False
            self.param_save_flag_list.clear()
            if self.save_top_k is not None:
                if self.parallel_dump:
                    if path != self.path+'-last':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.save_top_k:
                        shutil.rmtree(self.path_list_[0])
                        del self.path_list_[0]
                else:
                    if path != self.path+'-last.dat':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.save_top_k:
                        os.remove(self.path_list_[0])
                        del self.path_list_[0]
            else:
                if self.parallel_dump:
                    if path != self.path+'-last':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.max_save_files:
                        shutil.rmtree(self.path_list_[0])
                        del self.path_list_[0]
                else:
                    if path != self.path+'-last.dat':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.max_save_files:
                        os.remove(self.path_list_[0])
                        del self.path_list_[0]
        if self.parallel_training_and_save and hasattr(self, 'param_'):
            if self.parallel_dump==True:
                counter=0
                for i in range(len(self.param)):
                    if type(self.param[i])==list:
                        for j in range(len(self.param[i])):
                            counter+=1
                            process=mp.Process(target=self.parallel_param_dump,args=(self.param[i][j], i, j, path, counter))
                            process.start()
                    else:
                        counter+=1
                        process=mp.Process(target=self.parallel_param_dump,args=(self.param[i], i, None, path, counter))
                        process.start()
            else:
                output_file=open(path,'wb')
                pickle.dump(self.param_,output_file)
                output_file.close()
        else:
            output_file=open(path,'wb')
            pickle.dump(self.param,output_file)
            output_file.close()
        if self.parallel_training_and_save:
            self.save_flag.value=True
        return
    
    
    def restore_param(self,path):
        if self.parallel_dump==True:
            manager=mp.Manager()
            param=manager.list()
            counter=0
            for i in range(len(self.param)):
                if type(self.param[i])==list:
                    param.append(manager.list([None for _ in range(len(self.param[i]))]))
                else:
                    param.append(None)
            process_list=[]
            for i in range(len(self.param)):
                if type(self.param[i])==list:
                    for j in range(len(self.param[i])):
                        counter+=1
                        input_file1=open(os.path.join(path,"param_index_{counter}.dat"),'rb')
                        param_index=pickle.load(input_file1)
                        process=mp.Process(target=self.parallel_param_load,args=(param, param_index, path, counter))
                        process.start()
                        process_list.append(process)
                        input_file1.close()
                else:
                    counter+=1
                    input_file1=open(os.path.join(path,"param_index_{counter}.dat"),'rb')
                    param_index=pickle.load(input_file1)
                    process=mp.Process(target=self.parallel_state_load,args=(param, param_index, path, counter))
                    process.start()
                    process_list.append(process)
                    input_file1.close()
            for process in process_list:
                process.join()
            nn.assign_param(self.param,param)
        else:
            input_file=open(path,'rb')
            param=pickle.load(input_file)
            nn.assign_param(self.param,param)
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
        if self.save_top_k is None:
            if self.max_save_files==1:
                path=path
            else:
                if self.avg_reward!=None:
                    path=path.replace(path[path.find('.'):],'-{0:.4f}.dat'.format(self.avg_reward))
            self.path_list.append(path)
            if len(self.path_list)>self.max_save_files:
                os.remove(self.path_list[0])
                del self.path_list[0]
            self.save(path)
        else:
            if len(self.reward_list)>=self.trial_count:
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                if self.best_avg_reward==None:
                    self.best_avg_reward=avg_reward
                elif avg_reward>self.best_avg_reward:
                    self.path_list.append(path)
                    if len(self.path_list)>self.save_top_k:
                        os.remove(self.path_list[0])
                        del self.path_list[0]
                    self.best_avg_reward=avg_reward
                    self.save(path)
        if self.save_last:
            path=self.path+'-last.dat'
            self.save(path)
        if self.pool_network and not self.save_data:
            for i in range(self.processes):
                self.state_pool_list[i]=state_pool_list[i]
                self.action_pool_list[i]=action_pool_list[i]
                self.next_state_pool_list[i]=next_state_pool_list[i]
                self.reward_pool_list[i]=reward_pool_list[i]
                self.done_pool_list[i]=done_pool_list[i]
        return
    
    
    def _save(self,path):
        if self.save_top_k is not None:
            if path != self.path+'-last.dat':
                self.path_list.append(path)
            if len(self.path_list)>self.save_top_k:
                os.remove(self.path_list[0])
                del self.path_list[0]
        else:
            if path != self.path+'-last.dat':
                self.path_list.append(path)
            if len(self.path_list)>self.max_save_files:
                os.remove(self.path_list[0])
                del self.path_list[0]
        output_file=open(path,'wb')
        param=self.param
        self.param=None
        if type(self.optimizer)==list:
            opt_config=[opt.get_config() for opt in self.optimizer]
        else:
            opt_config=self.optimizer.get_config()
        self.opt_config=opt_config
        optimizer=self.optimizer
        self.optimizer=None
        pickle.dump(self,output_file)
        self.param=param
        self.optimizer=optimizer
        output_file.close()
        return
    
    
    def parallel_param_dump(self, param, index1, index2, path, counter):
        self.param_save_flag_list.append(False)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"param_{counter}.dat")
        output_file=open(filename,'wb')
        if type(param)==list:
            pickle.dump(param,output_file)
            output_file.close()
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"param_index_{counter}.dat")
            output_file=open(path,'wb')
            pickle.dump((index1, index2),output_file)
            output_file.close()
        else:
            pickle.dump(param,output_file)
            output_file.close()
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"param_index_{counter}.dat")
            output_file=open(path,'wb')
            pickle.dump((index1, index2),output_file)
            output_file.close()
        self.param_save_flag_list[counter]=True
            
    
    def parallel_state_dump(self, state_dict, index1, index2, path, counter):
        self.state_save_flag_list.append(False)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"state_{counter}.dat")
        output_file=open(path,'wb')
        if type(self.optimizer)==list:
            pickle.dump(state_dict,output_file)
            output_file.close()
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"state_index_{counter}.dat")
            output_file=open(path,'wb')
            pickle.dump((index1, str(index2)),output_file)
            output_file.close()
        else:
            pickle.dump(state_dict,output_file)
            output_file.close()
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"state_index_{counter}.dat")
            output_file=open(path,'wb')
            pickle.dump(str(index2),output_file)
            output_file.close()
        self.state_save_flag_list=True
    
    
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
        if self.parallel_training_and_save:
            self.save_flag.value=False
            if self.save_top_k is not None:
                if self.parallel_dump:
                    if path != self.path+'-last':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.save_top_k:
                        shutil.rmtree(self.path_list_[0])
                        del self.path_list_[0]
                else:
                    if path != self.path+'-last.dat':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.save_top_k:
                        os.remove(self.path_list_[0])
                        del self.path_list_[0]
            else:
                if self.parallel_dump:
                    if path != self.path+'-last':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.max_save_files:
                        shutil.rmtree(self.path_list_[0])
                        del self.path_list_[0]
                else:
                    if path != self.path+'-last.dat':
                        self.path_list_.append(path)
                    if len(self.path_list_)>self.max_save_files:
                        os.remove(self.path_list_[0])
                        del self.path_list_[0]
        else:
            output_file=open(path,'wb')
            param=self.param
            self.param=None
            if type(self.optimizer)==list:
                opt_config=[opt.get_config() for opt in self.optimizer]
            else:
                opt_config=self.optimizer.get_config()
            self.opt_config=opt_config
            optimizer=self.optimizer
            self.optimizer=None
            pickle.dump(self,output_file)
        if self.parallel_training_and_save:
            if self.parallel_dump==True:
                counter=0
                for i in range(len(self.param)):
                    if type(self.param[i])==list:
                        for j in range(len(self.param_[i])):
                            counter+=1
                            process=mp.Process(target=self.parallel_param_dump,args=(self.param[i][j], i, j, path, counter))
                            process.start()
                    else:
                        counter+=1
                        process=mp.Process(target=self.parallel_param_dump,args=(self.param[i], i, None, path, counter))
                        process.start()
            else:
                output_file=open(path,'wb')
                pickle.dump(self.param_,output_file)
        else:
            pickle.dump(param,output_file)
            self.param=param
            self.optimizer=optimizer
        if self.parallel_training_and_save:
            if self.parallel_dump==True:
                counter=0
                if type(self.optimizer)==list:
                    for i in range(len(self.optimizer)):
                        for j in range(len(self.state_dict[i])):
                            counter+=1
                            process=mp.Process(target=self.parallel_state_dump,args=(self.state_dict[i][str(j)], i, j, path, counter))
                            process.start()
                else:
                    for i in range(len(self.state_dict)):
                        counter+=1
                        process=mp.Process(target=self.parallel_state_dump,args=(self.state_dict[str(i)], i, None, path, counter))
                        process.start()
            else:
                pickle.dump(self.state_dict,output_file)
                output_file.close()
        else:
            if type(self.optimizer)==list:
                state_dict=[]
                for i in range(len(self.optimizer)):
                    state_dict.append(dict())
                    self.optimizer[i].save_own_variables(state_dict[-1])
                pickle.dump(state_dict,output_file)
            else:
                state_dict=dict()
                self.optimizer.save_own_variables(state_dict)
                pickle.dump(state_dict,output_file)
            output_file.close()
        if self.pool_network and not self.save_data:
            for i in range(self.processes):
                self.state_pool_list[i]=state_pool_list[i]
                self.action_pool_list[i]=action_pool_list[i]
                self.next_state_pool_list[i]=next_state_pool_list[i]
                self.reward_pool_list[i]=reward_pool_list[i]
                self.done_pool_list[i]=done_pool_list[i]
        return
    
    
    def save_in_parallel(self, path):
        if self.save_param_only==False:
            if type(self.optimizer)==list:
                self.state_dict=[]
                for i in range(len(self.optimizer)):
                    self.state_dict.append(dict())
                    self.optimizer[i].save_own_variables(self.state_dict[-1])
            else:
                self.state_dict=dict()
                self.optimizer.save_own_variables(self.state_dict)
            if self.parallel_dump:
                self._save(path+'.dat')
                process=mp.Process(target=self.save,args=(path,))
                process.start()
            else:
                self._save(path)
                process=mp.Process(target=self.save,args=(path.replace(path[self.path.find('.'):],'-parallel.dat'),))
                process.start()
        else:
            if self.parallel_training_and_save:
                if self.parallel_dump:
                    process=mp.Process(target=self.save_param,args=(path,))
                    process.start()
                else:
                    process=mp.Process(target=self.save_param,args=(self.path.replace(self.path[self.path.find('.'):],'-parallel.dat'),))
                    process.start()
    
    
    def save_checkpoint(self):
        if self.parallel_dump:
            if self.save_freq!=None:
                path=self.path+'-{0}'.format(self.total_epoch)
            elif self.save_freq_!=None:
                path=self.path+'-{0}'.format(self.batch_counter)
        else:
            if self.save_freq!=None:
                path=self.path+'-{0}.dat'.format(self.total_epoch)
            elif self.save_freq_!=None:
                path=self.path+'-{0}.dat'.format(self.batch_counter)
        if self.save_param_only==False:
            if self.parallel_training_and_save:
                if self.save_top_k is not None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.best_avg_reward==None:
                            self.best_avg_reward=avg_reward
                        elif avg_reward>self.best_avg_reward:
                            self.best_avg_reward=avg_reward
                            if self.parallel_dump:
                                path=path+'-{0:.4f}'.format(self.avg_reward)
                                path=path+'-best'
                            else:
                                path=path.replace(path[path.find('.'):],'-{0:.4f}.dat'.format(self.avg_reward))
                                path=path.replace(path[path.find('.'):],'-best.dat')
                            self.save_in_parallel(path)
                else:
                    if self.parallel_dump:
                        if self.avg_reward!=None:
                            path=path+'-{0:.4f}'.format(self.avg_reward)
                    else:
                        if self.avg_reward!=None:
                            path=path.replace(path[path.find('.'):],'-{0:.4f}.dat'.format(self.avg_reward))
                    self.save_in_parallel(path)
                if self.save_last:
                    if self.parallel_dump:
                        path=self.path+'-last'
                        self._save(path+'.dat')
                    else:
                        path=self.path+'-last.dat'
                        self._save(path)
                    self.save_in_parallel(path)
            else:
                self.save_(path)
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        model=pickle.load(input_file)
        param=self.param
        self.__dict__.update(model.__dict__)
        self.param=param
        param=pickle.load(input_file)
        nn.assign_param(self.param,param)
        if type(self.optimizer)==list:
            state_dict=pickle.load(input_file)
            for i in range(len(self.optimizer)):
                self.optimizer[i].from_config(self.opt_config[i])
                self.optimizer[i].built=False
                self.optimizer[i].build(self.optimizer[i]._trainable_variables)
                self.optimizer[i].load_own_variables(state_dict[i])
        else:
            state_dict=pickle.load(input_file)
            self.optimizer.from_config(self.opt_config)
            self.optimizer.built=False
            self.optimizer.build(self.optimizer._trainable_variables)
            self.optimizer.load_own_variables(state_dict)
        input_file.close()
        return
    
    
    def parallel_param_load(self, param, param_index, path, counter):
        input_file2=open(os.path.join(path,f"param_{counter}.dat"),'rb')
        if type(param[param_index[0]])==list:
            param[param_index[0]][param_index[1]]=pickle.load(input_file2)
            input_file2.close()
        else:
            param[param_index]=pickle.load(input_file2)
            input_file2.close()
            
    
    def parallel_state_load(self, state_dict, state_index, path, counter):
        input_file2=open(os.path.join(path,f"state_{counter}.dat"),'rb')
        if type(self.optimizer)==list:
            state_dict[state_index[0]][self.state_index[1]]=pickle.load(input_file2)
            input_file2.close()
        else:
            state_dict[state_index]=pickle.load(input_file2)
            input_file2.close()
    
    
    def restore_p(self,path1,path2):
        input_file1=open(path1,'rb')
        if not self.parallel_dump:
            input_file2=open(path2,'rb')
        model=pickle.load(input_file1)
        param=self.param
        self.__dict__.update(model.__dict__)
        if self.parallel_dump==True:
            manager=mp.Manager()
            param=manager.list()
            counter=0
            for i in range(len(self.param)):
                if type(self.param[i])==list:
                    param.append(manager.list([None for _ in range(len(self.param[i]))]))
                else:
                    param.append(None)
            process_list=[]
            for i in range(len(self.param)):
                if type(self.param[i])==list:
                    for j in range(len(self.param[i])):
                        counter+=1
                        input_file3=open(os.path.join(path2,"param_index_{counter}.dat"),'rb')
                        param_index=pickle.load(input_file3)
                        process=mp.Process(target=self.parallel_param_load,args=(param, param_index, path2, counter))
                        process.start()
                        process_list.append(process)
                        input_file3.close()
                else:
                    counter+=1
                    input_file3=open(os.path.join(path2,"param_index_{counter}.dat"),'rb')
                    param_index=pickle.load(input_file3)
                    process=mp.Process(target=self.parallel_state_load,args=(param, param_index, path2, counter))
                    process.start()
                    process_list.append(process)
                    input_file3.close()
        else:
            self.param=param
            param=pickle.load(input_file2)
            nn.assign_param(self.param,param)
        if self.parallel_dump==True:
            counter=0
            if type(self.optimizer)==list:
                state_dict=manager.list()
                for i in range(len(self.optimizer)):
                    state_dict.append(manager.dict())
                for i in range(len(self.optimizer)):
                    for j in range(len(self.state_dict[i])):
                        counter+=1
                        input_file3=open(os.path.join(path2,"state_index_{counter}.dat"),'rb')
                        state_index=pickle.load(input_file3)
                        process=mp.Process(target=self.parallel_state_load,args=(state_dict, state_index, path2, counter))
                        process.start()
                        process_list.append(process)
                    input_file3.close()
            else:
                state_dict=manager.dict()
                for i in range(len(self.state_dict)):
                    counter+=1
                    input_file3=open(os.path.join(path2,"state_index_{counter}.dat"),'rb')
                    state_index=pickle.load(input_file3)
                    process=mp.Process(target=self.parallel_param_load,args=(state_dict, state_index, path2, counter))
                    process.start()
                    process_list.append(process)
                input_file3.close()
        else:
            if type(self.optimizer)==list:
                state_dict=pickle.load(input_file2)
                for i in range(len(self.optimizer)):
                    self.optimizer[i].from_config(self.opt_config[i])
                    self.optimizer[i].built=False
                    self.optimizer[i].build(self.optimizer[i]._trainable_variables)
                    self.optimizer[i].load_own_variables(state_dict[i])
            else:
                state_dict=pickle.load(input_file2)
                self.optimizer.from_config(self.opt_config)
                self.optimizer.built=False
                self.optimizer.build(self.optimizer._trainable_variables)
                self.optimizer.load_own_variables(state_dict)
        input_file1.close()
        if not self.parallel_dump:
            input_file2.close()
        else:
            for process in process_list:
                process.join()
            nn.assign_param(self.param,param)
            counter=0
            if type(self.optimizer)==list:
                for i in range(len(self.optimizer)):
                    for j in range(len(self.state_dict[i])):
                        counter+=1
                        input_file3=open(os.path.join(path2,"state_index_{counter}.dat"),'rb')
                        state_index=pickle.load(input_file3)
                    self.optimizer[state_index[0]].from_config(self.opt_config[state_index[0]])
                    self.optimizer[state_index[0]].built=False
                    self.optimizer[state_index[0]].build(self.optimizer[state_index[0]]._trainable_variables)
                    self.optimizer[state_index[0]].load_own_variables(state_dict[state_index[0]])
                    input_file3.close()
            else:
                self.optimizer.from_config(self.opt_config)
                self.optimizer.built=False
                self.optimizer.build(self.optimizer._trainable_variables)
                self.optimizer.load_own_variables(state_dict)
                input_file3.close()
        return
