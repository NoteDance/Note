import tensorflow as tf
from Note import nn
from tensorflow.python.util import nest
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
        self.seed=7
        self.path=None
        self.save_freq=1
        self.save_freq_=None
        self.max_save_files=None
        self.save_best_only=False
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
    
    
    def set(self,policy=None,noise=None,pool_size=None,batch=None,num_updates=None,num_steps=None,update_batches=None,update_steps=None,trial_count=None,criterion=None,PPO=False,HER=False,MARL=False,PR=False,IRL=False,initial_ratio=1.0,initial_TD=7.,lambda_=0.5,alpha=0.7,max_batch=None):
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
        self.initial_ratio=np.array(initial_ratio).astype('float32')
        self.initial_TD=np.array(initial_TD).astype('float32')
        if PR:
            if PPO:
                self.prioritized_replay.PPO=PPO
                self.prioritized_replay.ratio=self.initial_ratio
                self.prioritized_replay.TD=self.initial_TD
                if max_batch is not None:
                    self.prioritized_replay.ratio_=tf.Variable(tf.zeros([max_batch]))
                    self.prioritized_replay.TD_=tf.Variable(tf.zeros([max_batch]))
                else:
                    self.prioritized_replay.ratio_=tf.Variable(tf.zeros([batch]))
                    self.prioritized_replay.TD_=tf.Variable(tf.zeros([batch]))
                self.prioritized_replay.batch=tf.Variable(tf.zeros((),dtype=tf.int32))
            else:
                self.prioritized_replay.TD=self.initial_TD
                if max_batch is not None:
                    self.prioritized_replay.TD_=tf.Variable(tf.zeros([max_batch]))
                else:
                    self.prioritized_replay.TD_=tf.Variable(tf.zeros([batch]))
                self.prioritized_replay.batch=tf.Variable(tf.zeros((),dtype=tf.int32))
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
        if self.jit_compile==True:
            output=self.forward(s,i)
        else:
            output=self.forward_(s,i)
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
                a=self.policy.select_action(output)
            elif isinstance(policy, rl.AdaptiveEpsGreedyPolicy):
                a=self.policy.select_action(output, self.step_counter)
            elif isinstance(policy, rl.GreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(policy, rl.BoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(policy, rl.MaxBoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(policy, rl.BoltzmannGumbelQPolicy):
                a=self.policy.select_action(output, self.step_counter)
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
    
    
    def compute_ess_from_weights(self, weights):
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
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
    
    
    def adjust_batch(self, batch_params, ema, target):
        if self.processes_her==None and self.processes_pr==None:
            buf_len = len(self.state_pool)
        else:
            buf_len = len(self.state_pool[7])
        if batch_params['min'] is None:
            cur_batch = self.batch
            batch_params['min'] = max(1, cur_batch // 2)
        if batch_params['max'] is None:
            batch_params['max'] = max(1, buf_len)
            
        if target != None:
            batch = int(round(self.batch * ema / target * float(batch_params['scale'])))
        else:
            batch = int(round(ema * float(batch_params['scale'])))
        batch = int(np.clip(batch, batch_params['min'], batch_params['max']))
        
        if batch_params['align'] is None:
            batch_params['align'] = self.batch
        new_batch = batch_params['align'] * (batch // batch_params['align'])
        self.batch = new_batch
    
    
    def adjust_alpha(self, alpha_params, ema, target, GNS=False):
        if not GNS:
            target_alpha = self.alpha + alpha_params['rate'] * (ema / target - 1.0)
        else:
            target_alpha = self.alpha + alpha_params['rate'] * (target - ema) / target
        target_alpha = np.clip(target_alpha, alpha_params['min'], alpha_params['max'])
        smooth = alpha_params.get('smooth', 0.2)
        self.alpha = smooth * self.alpha + (1.0 - smooth) * target_alpha
        self.alpha = float(self.alpha)
    
    
    def adjust_lr(self, lr_params, lr, ema, target, GNS=False): 
        if not GNS:
            target_lr = lr + lr_params['rate'] * (ema / target - 1.0)
        else:
            target_lr = lr + lr_params['rate'] * (target - ema) / target
        target_lr = np.clip(target_lr, lr_params['min'], lr_params['max'])
        smooth = lr_params.get('smooth', 0.2)
        lr = smooth * lr + (1.0 - smooth) * target_lr
        return lr
    
    
    def adjust_eps(self, eps_params, eps, ema, target, GNS=False):
        if not GNS:
            target_eps = eps + eps_params['rate'] * (target - ema) / target
        else:
            target_eps = eps + eps_params['rate'] * (ema / target - 1.0)
        target_eps = np.clip(target_eps, eps_params['min'], eps_params['max'])
        smooth = eps_params.get('smooth', 0.2)
        eps = smooth * eps + (1.0 - smooth) * target_eps
        return float(eps)
    
    
    def adjust_update_freq(self, freq_params, ema, target, GNS=False):
        if self.update_batches is not None:
            freq = self.update_batches
        else:
            freq = self.update_steps
        if not GNS:
            freq = freq_params['scale'] * target / ema * freq
        else:
            freq = freq_params['scale'] * ema / target * freq
        freq = np.clip(freq, freq_params['min'], freq_params['max'])
        if self.update_batches is not None:
            self.update_batches = int(freq)
        else:
            self.update_steps = int(freq)


    def adjust_tau(self, tau_params, ema, target, GNS=False):      
        tau = self.tau
        if not GNS:
            target_tau = tau + tau_params['rate'] * (ema / target - 1.0)
        else:
            target_tau = tau + tau_params['rate'] * (target - ema) / target
        target_tau = np.clip(target_tau, tau_params['min'], tau_params['max'])
        smooth = tau_params.get('smooth', 0.2)
        tau = smooth * tau + (1.0 - smooth) * target_tau
        self.tau = float(tau)
    
    
    def adjust_gamma(self, gamma_params, ema, target, GNS=False):    
        gamma = self.gamma
        if not GNS:
            target_gamma = gamma + gamma_params['rate'] * (ema / target - 1.0)
        else:
            target_gamma = gamma + gamma_params['rate'] * (target - ema) / target
        target_gamma = np.clip(target_gamma, gamma_params['min'], gamma_params['max'])
        smooth = gamma_params.get('smooth', 0.2)
        gamma = smooth * gamma + (1.0 - smooth) * target_gamma
        self.gamma.assign(gamma)
        
        
    def adjust_num_store(self, store_params, ema, target):      
        num_store = store_params['scale'] * target / ema * self.num_store
        num_store = np.clip(num_store, store_params['min'], store_params['max'])
        self.num_store = int(max(store_params['min'], num_store))
    
    
    def adjust_weight_decay(self, weight_decay_params, weight_decay, ema, target, GNS=False):
        if not GNS:
            target_weight_decay = weight_decay + weight_decay_params['rate'] * (target - ema) / target
        else:
            target_weight_decay = weight_decay + weight_decay_params['rate'] * (ema / target - 1.0)
        target_weight_decay = np.clip(target_weight_decay, weight_decay_params['min'], weight_decay_params['max'])
        smooth = weight_decay_params.get('smooth', 0.2)
        weight_decay = smooth * weight_decay + (1.0 - smooth) * target_weight_decay
        return weight_decay
    
    
    def adjust_beta1(self, beta1_params, beta1, ema, target, GNS=False): 
        target_beta1 = beta1 + beta1_params['rate'] * (ema / target - 1.0)
        target_beta1 = np.clip(target_beta1, beta1_params['min'], beta1_params['max'])
        smooth = beta1_params.get('smooth', 0.2)
        beta1 = smooth * beta1 + (1.0 - smooth) * target_beta1
        return beta1
    
    
    def adjust_beta2(self, beta2_params, beta2, ema, target, GNS=False):
        target_beta2 = beta2 + beta2_params['rate'] * (ema / target - 1.0)
        target_beta2 = np.clip(target_beta2, beta2_params['min'], beta2_params['max'])
        smooth = beta2_params.get('smooth', 0.2)
        beta2 = smooth * beta2 + (1.0 - smooth) * target_beta2
        return beta2
    
    
    def adjust_clip(self, clip_params, ema, target, GNS=False):      
        clip = self.clip
        if not GNS:
            target_clip = clip + clip_params['rate'] * (ema / target - 1.0)
        else:
            target_clip = clip + clip_params['rate'] * (target - ema) / target
        target_clip = np.clip(target_clip, clip_params['min'], clip_params['max'])
        smooth = clip_params.get('smooth', 0.2)
        clip = smooth * clip + (1.0 - smooth) * target_clip
        self.clip.assign(clip)
        
        
    def adjust_beta(self, beta_params, ema, target, GNS=False):
        beta = self.beta
        if not GNS:
            target_beta = beta + beta_params['rate'] * (target - ema) / target
        else:
            target_beta = beta + beta_params['rate'] * (ema / target - 1.0)
        target_beta = np.clip(target_beta, beta_params['min'], beta_params['max'])
        smooth = beta_params.get('smooth', 0.2)
        beta = smooth * beta + (1.0 - smooth) * target_beta
        self.beta.assign(beta)
    
    
    def compute_ess(self, ema_ess, smooth_alpha):
        if self.PPO:
            scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
            weights = tf.pow(scores + 1e-7, self.alpha)
        else:
            weights = tf.pow(self.prioritized_replay.TD + 1e-7, self.alpha)
            
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        
        if ema_ess is None:
            ema = ess
        else:
            ema = smooth_alpha * ess + (1.0 - smooth_alpha) * ema_ess
            
        return float(ema)
    
    
    def adjust_batch_size(self, smooth_alpha=0.2, batch_params=None, target_ess=None, alpha_params=None, lr_params=None, eps_params=None, freq_params=None, tau_params=None, gamma_params=None, store_params=None, weight_decay_params=None, beta1_params=None, beta2_params=None, clip_params=None, beta_params=None):
        if not hasattr(self, 'ema_ess'):
            self.ema_ess = None
        
        ema = self.compute_ess(self.ema_ess, smooth_alpha)
        
        if batch_params is not None:
            self.adjust_batch(batch_params, ema, target_ess)
        
        if alpha_params is not None and target_ess is not None:
            self.adjust_alpha(alpha_params, ema, target_ess)
            
        if lr_params is not None and target_ess is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    optimizer.learning_rate.assign(self.adjust_lr(lr_params, optimizer.learning_rate, ema, target_ess))
                    if hasattr(optimizer, 'adamw_lr'):
                        optimizer.adamw_lr.assign(self.adjust_lr(lr_params, optimizer.adamw_lr, ema, target_ess))
            else:
                self.optimizer.learning_rate.assign(self.adjust_lr(lr_params, self.optimizer.learning_rate, ema, target_ess))
                if hasattr(optimizer, 'adamw_lr'):
                    self.optimizer.adamw_lr.assign(self.adjust_lr(lr_params, self.optimizer.adamw_lr, ema, target_ess))
    
        if eps_params is not None and target_ess is not None:
            if type(self.policy) == list:
                for policy in self.policy:
                    policy.eps = self.adjust_eps(eps_params, policy.eps, ema, target_ess)
            else:
                self.policy.eps = self.adjust_eps(eps_params, self.policy.eps, ema, target_ess)
                
        if freq_params is not None and target_ess is not None:
            self.adjust_update_freq(freq_params, ema, target_ess)
        
        if tau_params is not None and target_ess is not None:
            self.adjust_tau(tau_params, ema, target_ess)
            
        if gamma_params is not None and target_ess is not None:
            self.adjust_gamma(gamma_params, ema, target_ess)
        
        if store_params is not None and target_ess is not None:
            self.adjust_num_store(store_params, ema, target_ess)
            
        if weight_decay_params is not None and target_ess is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    optimizer.weight_decay.assign(self.adjust_weight_decay(weight_decay_params, optimizer.weight_decay, ema, target_ess))
                    if hasattr(optimizer, 'adamw_wd'):
                        optimizer.adamw_wd.assign(self.adjust_weight_decay(weight_decay_params, optimizer.adamw_wd, ema, target_ess))
            else:
                self.optimizer.weight_decay.assign(self.adjust_weight_decay(weight_decay_params, self.optimizer.weight_decay, ema, target_ess))
                if hasattr(optimizer, 'adamw_wd'):
                    self.optimizer.adamw_wd.assign(self.adjust_weight_decay(weight_decay_params, self.optimizer.adamw_wd, ema, target_ess))
        
        if beta1_params is not None and target_ess is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    if not hasattr(optimizer, 'betas'):
                        optimizer.beta1.assign(self.adjust_beta1(beta1_params, optimizer.beta1, ema, target_ess))
                    else:
                        optimizer.betas[0].assign(self.adjust_beta1(beta1_params, optimizer.betas[0], ema, target_ess))
                    if hasattr(optimizer, 'adamw_betas'):
                        optimizer.adamw_betas[0].assign(self.adjust_beta1(beta1_params, optimizer.adamw_betas[0], ema, target_ess))
            else:
                if not hasattr(optimizer, 'betas'):
                    self.optimizer.beta1.assign(self.adjust_beta1(beta1_params, self.optimizer.beta1, ema, target_ess))
                else:
                    self.optimizer.betas[0].assign(self.adjust_beta1(beta1_params, self.optimizer.betas[0], ema, target_ess))
                if hasattr(optimizer, 'adamw_betas'):
                    self.optimizer.adamw_betas[0].assign(self.adjust_beta1(beta1_params, self.optimizer.adamw_betas[0], ema, target_ess))
        
        if beta2_params is not None and target_ess is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    if not hasattr(optimizer, 'betas'):
                        optimizer.beta2.assign(self.adjust_beta2(beta2_params, optimizer.beta2, ema, target_ess))
                    else:
                        optimizer.betas[1].assign(self.adjust_beta2(beta2_params, optimizer.betas[1], ema, target_ess))
                    if hasattr(optimizer, 'adamw_betas'):
                        optimizer.adamw_betas[1].assign(self.adjust_beta2(beta2_params, optimizer.adamw_betas[1], ema, target_ess))
            else:
                if not hasattr(optimizer, 'betas'):
                    self.optimizer.beta2.assign(self.adjust_beta1(beta2_params, self.optimizer.beta2, ema, target_ess))
                else:
                    self.optimizer.betas[1].assign(self.adjust_beta2(beta2_params, self.optimizer.betas[1], ema, target_ess))
                if hasattr(optimizer, 'adamw_betas'):
                    self.optimizer.adamw_betas[1].assign(self.adjust_beta2(beta2_params, self.optimizer.adamw_betas[1], ema, target_ess))
        
        if clip_params is not None and target_ess is not None:
            self.adjust_clip(clip_params, ema, target_ess)
        
        if beta_params is not None and target_ess is not None:
            self.adjust_beta(beta_params, ema, target_ess)
    
    
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
    
    
    def estimate_gradient_variance(self, batch_size, num_samples, ema_noise, smooth_alpha, jit_compile=True):
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
            ema_noise = smooth_alpha * variance + (1 - smooth_alpha) * ema_noise
            
        return ema_noise
    
    
    def adabatch(self, num_samples, target_noise=1e-3, smooth_alpha=0.2, batch_params=None, alpha_params=None, lr_params=None, eps_params=None, freq_params=None, tau_params=None, gamma_params=None, weight_decay_params=None, beta1_params=None, beta2_params=None, clip_params=None, beta_params=None, jit_compile=True):
        if not hasattr(self, 'ema_noise'):
            self.ema_noise = None
        
        ema_noise = self.estimate_gradient_variance(self.batch, num_samples, self.ema_noise, smooth_alpha, jit_compile)
        
        if batch_params is not None:
            self.adjust_batch(batch_params, ema_noise, target_noise)
        
        if alpha_params is not None:
            self.adjust_alpha(alpha_params, ema_noise, target_noise, True)
            
        if lr_params is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    optimizer.learning_rate.assign(self.adjust_lr(lr_params, optimizer.learning_rate, ema_noise, target_noise, True))
                    if hasattr(optimizer, 'adamw_lr'):
                        optimizer.adamw_lr.assign(self.adjust_lr(lr_params, optimizer.adamw_lr, ema_noise, target_noise, True))
            else:
                self.optimizer.learning_rate.assign(self.adjust_lr(lr_params, self.optimizer.learning_rate, ema_noise, target_noise, True))
                if hasattr(optimizer, 'adamw_lr'):
                    self.optimizer.adamw_lr.assign(self.adjust_lr(lr_params, self.optimizer.adamw_lr, ema_noise, target_noise, True))
    
        if eps_params is not None:
            if type(self.policy) == list:
                for policy in self.policy:
                    policy.eps = self.adjust_eps(eps_params, policy.eps, ema_noise, target_noise, True)
            else:
                self.policy.eps = self.adjust_eps(eps_params, self.policy.eps, ema_noise, target_noise, True)
            
        if freq_params is not None:
            self.adjust_update_freq(freq_params, ema_noise, target_noise, True)
        
        if tau_params is not None:
            self.adjust_tau(tau_params, ema_noise, target_noise, True)
            
        if gamma_params is not None:
            self.adjust_gamma(gamma_params, ema_noise, target_noise, True)
        
        if weight_decay_params is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    optimizer.weight_decay.assign(self.adjust_weight_decay(weight_decay_params, optimizer.weight_decay, ema_noise, target_noise, True))
                    if hasattr(optimizer, 'adamw_wd'):
                        optimizer.adamw_wd.assign(self.adjust_weight_decay(weight_decay_params, optimizer.adamw_wd, ema_noise, target_noise, True))
            else:
                self.optimizer.weight_decay.assign(self.adjust_weight_decay(weight_decay_params, self.optimizer.weight_decay, ema_noise, target_noise, True))
                if hasattr(optimizer, 'adamw_wd'):
                    self.optimizer.adamw_wd.assign(self.adjust_weight_decay(weight_decay_params, self.optimizer.adamw_wd, ema_noise, target_noise, True))
        
        if beta1_params is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    if not hasattr(optimizer, 'betas'):
                        optimizer.beta1.assign(self.adjust_beta1(beta1_params, optimizer.beta1, ema_noise, target_noise, True))
                    else:
                        optimizer.betas[0].assign(self.adjust_beta1(beta1_params, optimizer.betas[0], ema_noise, target_noise, True))
                    if hasattr(optimizer, 'adamw_betas'):
                        optimizer.adamw_betas[0].assign(self.adjust_beta1(beta1_params, optimizer.adamw_betas[0], ema_noise, target_noise, True))
            else:
                if not hasattr(optimizer, 'betas'):
                    self.optimizer.beta1.assign(self.adjust_beta1(beta1_params, self.optimizer.beta1, ema_noise, target_noise, True))
                else:
                    self.optimizer.betas[0].assign(self.adjust_beta1(beta1_params, self.optimizer.betas[0], ema_noise, target_noise, True))
                if hasattr(optimizer, 'adamw_betas'):
                    self.optimizer.adamw_betas[0].assign(self.adjust_beta1(beta1_params, self.optimizer.adamw_betas[0], ema_noise, target_noise, True))
        
        if beta2_params is not None:
            if type(self.optimizer) == list:
                for optimizer in self.optimizer:
                    if not hasattr(optimizer, 'betas'):
                        optimizer.beta2.assign(self.adjust_beta2(beta2_params, optimizer.beta2, ema_noise, target_noise, True))
                    else:
                        optimizer.betas[1].assign(self.adjust_beta2(beta2_params, optimizer.betas[1], ema_noise, target_noise, True))
                    if hasattr(optimizer, 'adamw_betas'):
                        optimizer.adamw_betas[1].assign(self.adjust_beta2(beta2_params, optimizer.adamw_betas[1], ema_noise, target_noise, True))
            else:
                if not hasattr(optimizer, 'betas'):
                    self.optimizer.beta2.assign(self.adjust_beta1(beta2_params, self.optimizer.beta2, ema_noise, target_noise, True))
                else:
                    self.optimizer.betas[1].assign(self.adjust_beta2(beta2_params, self.optimizer.betas[1], ema_noise, target_noise, True))
                if hasattr(optimizer, 'adamw_betas'):
                    self.optimizer.adamw_betas[1].assign(self.adjust_beta2(beta2_params, self.optimizer.adamw_betas[1], ema_noise, target_noise, True))
        
        if clip_params is not None:
            self.adjust_clip(clip_params, ema_noise, target_noise, True)
        
        if beta_params is not None:
            self.adjust_beta(beta_params, ema_noise, target_noise, True)
    
    
    def adjust(self, target_ess=None, target_noise=None, num_samples=None, smooth_alpha=0.2, batch_params=None, alpha_params=None, lr_params=None, eps_params=None, freq_params=None, tau_params=None, gamma_params=None, store_params=None, weight_decay_params=None, beta1_params=None, beta2_params=None, clip_params=None, beta_params=None, jit_compile=True):
        if target_noise is None:
            self.adjust_batch_size(smooth_alpha, batch_params, target_ess, alpha_params, lr_params, eps_params, freq_params, tau_params, gamma_params, store_params, weight_decay_params, beta1_params, beta2_params, clip_params, beta_params)
        else:
            self.adabatch(num_samples, target_noise, smooth_alpha, batch_params, alpha_params, lr_params, eps_params, freq_params, tau_params, gamma_params, weight_decay_params, beta1_params, beta2_params, clip_params, beta_params, jit_compile)
    
    
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
                s,a,next_s,r,d=self.prioritized_replay.sample(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.lambda_,self.alpha,self.batch)
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
    
    
    def dataset_fn(self, dataset, global_batch_size, input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
        dataset = dataset.batch(batch_size)
        return dataset
    
    
    @tf.function(jit_compile=True)
    def train_step(self, train_data, train_loss, optimizer):
        with tf.GradientTape() as tape:
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
        with tf.GradientTape() as tape:
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
        with tf.GradientTape() as tape:
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
        
        if self.PR==True or self.HER==True:
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
                    for p in range(self.processes):
                        if hasattr(self,'window_size_func'):
                            window_size=int(self.window_size_func(p))
                        if window_size!=None and len(self.state_pool_list[p])>window_size:
                            self.state_pool_list[p]=self.state_pool_list[p][window_size:]
                            self.action_pool_list[p]=self.action_pool_list[p][window_size:]
                            self.next_state_pool_list[p]=self.action_pool_list[p][window_size:]
                            self.reward_pool_list[p]=self.action_pool_list[p][window_size:]
                            self.done_pool_list[p]=self.action_pool_list[p][window_size:]
                            if self.PPO:
                                self.ratio_list[p]=self.ratio_list[p][window_size:]
                            self.TD_list[p]=self.TD_list[p][window_size:]
                    self.state_pool=np.concatenate(self.state_pool_list)
                    self.action_pool=np.concatenate(self.action_pool_list)
                    self.next_state_pool=np.concatenate(self.next_state_pool_list)
                    self.reward_pool=np.concatenate(self.reward_pool_list)
                    self.done_pool=np.concatenate(self.done_pool_list)
                    if self.PPO:
                        self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                        self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                    else:
                        self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
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
                                self.state_pool_list[p]=None
                                self.action_pool_list[p]=None
                                self.next_state_pool_list[p]=None
                                self.reward_pool_list[p]=None
                                self.done_pool_list[p]=None
                            break
                    if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                        self.adjust_func()
                        if self.num_updates!=None and self.batch_counter%self.update_batches==0:
                            self.state_pool=np.concatenate(self.state_pool_list)
                            self.action_pool=np.concatenate(self.action_pool_list)
                            self.next_state_pool=np.concatenate(self.next_state_pool_list)
                            self.reward_pool=np.concatenate(self.reward_pool_list)
                            self.done_pool=np.concatenate(self.done_pool_list)
                            self.pool_size_=self.num_updates*self.batch
                            if len(self.state_pool)>=self.pool_size_:
                                idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                            else:
                                idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                            self.state_pool=self.state_pool[idx]
                            self.action_pool=self.action_pool[idx]
                            self.next_state_pool=self.next_state_pool[idx]
                            self.reward_pool=self.reward_pool[idx]
                            self.done_pool=self.done_pool[idx]
                            train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).batch(self.batch)
                            multi_worker_dataset = self.strategy.distribute_datasets_from_function(
                                    lambda input_context: self.dataset_fn(train_ds, self.batch, input_context)) 
                if self.stop_training==True:
                    return total_loss,num_batches
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
        
        if self.PR==True or self.HER==True:
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
                    self.state_pool=np.concatenate(self.state_pool_list)
                    self.action_pool=np.concatenate(self.action_pool_list)
                    self.next_state_pool=np.concatenate(self.next_state_pool_list)
                    self.reward_pool=np.concatenate(self.reward_pool_list)
                    self.done_pool=np.concatenate(self.done_pool_list)
                    if self.PPO:
                        self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                        self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                    else:
                        self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
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
                                self.state_pool_list[p]=None
                                self.action_pool_list[p]=None
                                self.next_state_pool_list[p]=None
                                self.reward_pool_list[p]=None
                                self.done_pool_list[p]=None
                            break
                    if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                        self.adjust_func()
                        if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                            if self.num_updates!=None and self.batch_counter%self.update_batches==0:
                                self.state_pool=np.concatenate(self.state_pool_list)
                                self.action_pool=np.concatenate(self.action_pool_list)
                                self.next_state_pool=np.concatenate(self.next_state_pool_list)
                                self.reward_pool=np.concatenate(self.reward_pool_list)
                                self.done_pool=np.concatenate(self.done_pool_list)
                                self.pool_size_=self.num_updates*self.batch
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
                if self.stop_training==True:
                    coordinator.join()
                    return total_loss,num_batches
            coordinator.join()
            return total_loss,num_batches
    
    
    def train1(self, train_loss, optimizer):
        self.step_counter+=1
        batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
        if len(self.state_pool)%self.batch!=0:
            batches+=1
        if self.PR==True or self.HER==True:
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
                        return train_loss.result().numpy()
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    break
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_begin'):
                        callback.on_batch_begin(batch, logs={})
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                train_ds=tf.data.Dataset.from_tensor_slices((state_batch,action_batch,next_state_batch,reward_batch,done_batch)).batch(self.batch)
                if isinstance(self.strategy,tf.distribute.MirroredStrategy):
                    train_ds=self.strategy.experimental_distribute_dataset(train_ds)
                    for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                        if self.jit_compile==True:
                            loss=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.strategy)
                        else:
                            loss=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.strategy)
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
                                for p in range(self.processes):
                                    if hasattr(self,'window_size_func'):
                                        window_size=int(self.window_size_func(p))
                                    if window_size!=None and len(self.state_pool_list[p])>window_size:
                                        self.state_pool_list[p]=self.state_pool_list[p][window_size:]
                                        self.action_pool_list[p]=self.action_pool_list[p][window_size:]
                                        self.next_state_pool_list[p]=self.action_pool_list[p][window_size:]
                                        self.reward_pool_list[p]=self.action_pool_list[p][window_size:]
                                        self.done_pool_list[p]=self.action_pool_list[p][window_size:]
                                        if self.PPO:
                                            self.ratio_list[p]=self.ratio_list[p][window_size:]
                                        self.TD_list[p]=self.TD_list[p][window_size:]
                                self.state_pool=np.concatenate(self.state_pool_list)
                                self.action_pool=np.concatenate(self.action_pool_list)
                                self.next_state_pool=np.concatenate(self.next_state_pool_list)
                                self.reward_pool=np.concatenate(self.reward_pool_list)
                                self.done_pool=np.concatenate(self.done_pool_list)
                                if self.PPO:
                                    self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                                    self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                                    return (total_loss / num_batches).numpy()
                                else:
                                    self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                                self.adjust_func()
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
                    for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                        if self.jit_compile==True:
                            loss=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                        else:
                            loss=self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
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
                                for p in range(self.processes):
                                    if hasattr(self,'window_size_func'):
                                        window_size=int(self.window_size_func(p))
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
                                    return train_loss.result().numpy()
                                else:
                                    self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                                self.adjust_func()
                batch_logs = {'loss': loss.numpy()}
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_end'):
                        callback.on_batch_end(batch, logs=batch_logs)
                batch += 1
            if len(self.state_pool)%self.batch!=0:
                if self.stop_training==True:
                    if self.distributed_flag==True:
                        if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                            self.coordinator.join()
                            return total_loss.fetch() / num_batches
                        else:
                            return (total_loss / num_batches).numpy()
                    else:
                        return train_loss.result().numpy()
                if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                    if self.distributed_flag==True:
                        if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                            return total_loss.fetch() / num_batches
                        else:
                            return (total_loss / num_batches).numpy()
                    else:
                        return train_loss.result().numpy()
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_begin'):
                        callback.on_batch_begin(batch, logs={})
                state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                train_ds=tf.data.Dataset.from_tensor_slices((state_batch,action_batch,next_state_batch,reward_batch,done_batch)).batch(self.batch)
                if isinstance(self.strategy,tf.distribute.MirroredStrategy):
                    train_ds=self.strategy.experimental_distribute_dataset(train_ds)
                    for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                        if self.jit_compile==True:
                            loss=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.strategy)
                        else:
                            loss=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.strategy)
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
                                for p in range(self.processes):
                                    if hasattr(self,'window_size_func'):
                                        window_size=int(self.window_size_func(p))
                                    if window_size!=None and len(self.state_pool_list[p])>window_size:
                                        self.state_pool_list[p]=self.state_pool_list[p][window_size:]
                                        self.action_pool_list[p]=self.action_pool_list[p][window_size:]
                                        self.next_state_pool_list[p]=self.action_pool_list[p][window_size:]
                                        self.reward_pool_list[p]=self.action_pool_list[p][window_size:]
                                        self.done_pool_list[p]=self.action_pool_list[p][window_size:]
                                        if self.PPO:
                                            self.ratio_list[p]=self.ratio_list[p][window_size:]
                                        self.TD_list[p]=self.TD_list[p][window_size:]
                                self.state_pool=np.concatenate(self.state_pool_list)
                                self.action_pool=np.concatenate(self.action_pool_list)
                                self.next_state_pool=np.concatenate(self.next_state_pool_list)
                                self.reward_pool=np.concatenate(self.reward_pool_list)
                                self.done_pool=np.concatenate(self.done_pool_list)
                                if self.PPO:
                                    self.prioritized_replay.ratio=np.concat(self.ratio_list, axis=0)
                                    self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                                    return (total_loss / num_batches).numpy()
                                else:
                                    self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)
                            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                                self.adjust_func()
                elif isinstance(self.strategy,tf.distribute.MultiWorkerMirroredStrategy):
                    with self.strategy.scope():
                        multi_worker_dataset = self.strategy.distribute_datasets_from_function(
                            lambda input_context: self.dataset_fn(train_ds, self.batch_size, input_context))  
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
                        loss=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                    else:
                        loss=self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
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
                            for p in range(self.processes):
                                if hasattr(self,'window_size_func'):
                                    window_size=int(self.window_size_func(p))
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
                                return train_loss.result().numpy()
                            else:
                                self.prioritized_replay.TD=np.concat(self.TD_list, axis=0)  
                        if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                            self.adjust_func()
                if not isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                    batch_logs = {'loss': loss.numpy()}
                else:
                    batch_logs = {'loss': loss.fetch()}
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_end'):
                        callback.on_batch_end(batch, logs=batch_logs)
            if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                self.coordinator.join()
        else:
            batch = 0
            if self.distributed_flag==True:
                total_loss = 0.0
                num_batches = 0
                if self.pool_network==True:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).batch(self.batch)
                else:
                    if self.num_updates!=None:
                        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).batch(self.batch)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
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
                            loss=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.strategy)
                        else:
                            loss=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.strategy)
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
                                if self.PPO:
                                    for p in range(self.processes):
                                        self.state_pool_list[p]=None
                                        self.action_pool_list[p]=None
                                        self.next_state_pool_list[p]=None
                                        self.reward_pool_list[p]=None
                                        self.done_pool_list[p]=None
                                    break
                            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                                self.adjust_func()
                                if self.num_updates!=None and self.batch_counter%self.update_batches==0:
                                    self.state_pool=np.concatenate(self.state_pool_list)
                                    self.action_pool=np.concatenate(self.action_pool_list)
                                    self.next_state_pool=np.concatenate(self.next_state_pool_list)
                                    self.reward_pool=np.concatenate(self.reward_pool_list)
                                    self.done_pool=np.concatenate(self.done_pool_list)
                                    self.pool_size_=self.num_updates*self.batch
                                    if len(self.state_pool)>=self.pool_size_:
                                        idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                                    else:
                                        idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                                    self.state_pool=self.state_pool[idx]
                                    self.action_pool=self.action_pool[idx]
                                    self.next_state_pool=self.next_state_pool[idx]
                                    self.reward_pool=self.reward_pool[idx]
                                    self.done_pool=self.done_pool[idx]
                                    train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).batch(self.batch)
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
                if self.pool_network==True:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).batch(self.batch)
                else:
                    if self.num_updates!=None:
                        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).batch(self.batch)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                    if self.stop_training==True:
                        return train_loss.result().numpy() 
                    if self.num_updates!=None and self.batch_counter%self.num_updates==0:
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_batch_begin'):
                            callback.on_batch_begin(batch, logs={})
                    if self.jit_compile==True:
                        loss=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                    else:
                        loss=self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
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
                            if self.PPO:
                                for p in range(self.processes):
                                    self.state_pool_list[p]=None
                                    self.action_pool_list[p]=None
                                    self.next_state_pool_list[p]=None
                                    self.reward_pool_list[p]=None
                                    self.done_pool_list[p]=None
                                break
                        if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                            self.adjust_func()
                            if self.num_updates!=None and self.batch_counter%self.update_batches==0:
                                if self.processes_her==None and self.processes_pr==None:
                                    self.state_pool=np.concatenate(self.state_pool_list)
                                    self.action_pool=np.concatenate(self.action_pool_list)
                                    self.next_state_pool=np.concatenate(self.next_state_pool_list)
                                    self.reward_pool=np.concatenate(self.reward_pool_list)
                                    self.done_pool=np.concatenate(self.done_pool_list)
                                    self.pool_size_=self.num_updates*self.batch
                                    if len(self.state_pool)>=self.pool_size_:
                                        idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                                    else:
                                        idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                                    self.state_pool=self.state_pool[idx]
                                    self.action_pool=self.action_pool[idx]
                                    self.next_state_pool=self.next_state_pool[idx]
                                    self.reward_pool=self.reward_pool[idx]
                                    self.done_pool=self.done_pool[idx]
                                    train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).batch(self.batch)
                                else:
                                    self.state_pool[7]=np.concatenate(self.state_pool_list)
                                    self.action_pool[7]=np.concatenate(self.action_pool_list)
                                    self.next_state_pool[7]=np.concatenate(self.next_state_pool_list)
                                    self.reward_pool[7]=np.concatenate(self.reward_pool_list)
                                    self.done_pool[7]=np.concatenate(self.done_pool_list)
                                    self.pool_size_=self.num_updates*self.batch
                                    if len(self.state_pool)>=self.pool_size_:
                                        idx=np.random.choice(self.state_pool[7].shape[0], size=self.pool_size_, replace=False)
                                    else:
                                        idx=np.random.choice(self.state_pool[7].shape[0], size=self.state_pool[7].shape[0], replace=False)
                                    self.state_pool[7]=self.state_pool[7][idx]
                                    self.action_pool[7]=self.action_pool[7][idx]
                                    self.next_state_pool[7]=self.next_state_pool[7][idx]
                                    self.reward_pool[7]=self.reward_pool[7][idx]
                                    self.done_pool[7]=self.done_pool[7][idx]
        if self.update_steps!=None:
            if self.step_counter%self.update_steps==0:
                self.update_param()
                if self.PR:
                    if not hasattr(self,'window_size_func'):
                        if self.PPO:
                            window_size=self.window_size_ppo
                        else:
                            window_size=self.window_size_pr
                    if hasattr(self,'window_size_func'):
                        window_size=int(self.window_size_func())
                    if window_size!=None and len(self.state_pool)>window_size:
                        self.state_pool=self.state_pool[window_size:]
                        self.action_pool=self.action_pool[window_size:]
                        self.next_state_pool=self.action_pool[window_size:]
                        self.reward_pool=self.action_pool[window_size:]
                        self.done_pool=self.action_pool[window_size:]
                        if self.PPO:
                            self.prioritized_replay.ratio=self.prioritized_replay.ratio[window_size:]
                        self.prioritized_replay.TD=self.prioritized_replay.TD[window_size:]
                elif self.PPO:
                    self.state_pool=None
                    self.action_pool=None
                    self.next_state_pool=None
                    self.reward_pool=None
                    self.done_pool=None
                if self.PPO:
                    if self.distributed_flag==True:
                        if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                            return total_loss.fetch() / num_batches
                        else:
                            return (total_loss / num_batches).numpy()
                    else:
                        return train_loss.result().numpy()
            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                self.adjust_func()
                if self.step_counter%self.update_steps==0:
                    if self.num_updates!=None:
                        if len(self.state_pool)>=self.pool_size_:
                            idx=np.random.choice(self.state_pool.shape[0], size=self.pool_size_, replace=False)
                        else:
                            idx=np.random.choice(self.state_pool.shape[0], size=self.state_pool.shape[0], replace=False)
                        state_pool=self.state_pool[idx]
                        action_pool=self.action_pool[idx]
                        next_state_pool=self.action_pool[idx]
                        reward_pool=self.action_pool[idx]
                        done_pool=self.action_pool[idx]
                        train_ds=tf.data.Dataset.from_tensor_slices((state_pool,action_pool,next_state_pool,reward_pool,done_pool)).batch(self.batch)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
        else:
            self.update_param()
        if self.distributed_flag==True:
            if isinstance(self.strategy,tf.distribute.ParameterServerStrategy):
                return total_loss.fetch() / num_batches
            else:
                return (total_loss / num_batches).numpy()
        else:
            return train_loss.result().numpy()
    
    
    def train2(self, train_loss, optimizer):
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
            loss=self.train1(train_loss,optimizer)
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
                    inverse_len=tf.constant(self.inverse_len)
                    total_inverse=tf.reduce_sum(inverse_len)
                    prob=inverse_len/total_inverse
                    index=np.random.choice(self.processes,p=prob.numpy(),replace=False)
                    self.inverse_len[index]=1/(len(self.state_pool_list[index])+1)
            else:
                index=p
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
        counter=0
        while True:
            for p in range(self.processes):
                process=mp.Process(target=self.store_in_parallel,args=(p,lock_list))
                process.start()
                process_list.append(process)
            for process in process_list:
                process.join()
            counter+=1
            if self.state_pool is not None and len(self.state_pool)>=self.batch and counter<self.num_store:
                    continue
            if self.processes_her==None and self.processes_pr==None:
                self.state_pool=np.concatenate(self.state_pool_list)
                self.action_pool=np.concatenate(self.action_pool_list)
                self.next_state_pool=np.concatenate(self.next_state_pool_list)
                self.reward_pool=np.concatenate(self.reward_pool_list)
                self.done_pool=np.concatenate(self.done_pool_list)
                if counter<self.num_store and len(self.state_pool)<self.batch:
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
                if counter<self.num_store and len(self.state_pool[7])<self.batch:
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
            if hasattr(self, 'adjust_func') and len(self.state_pool)>=self.pool_size_:
                self.prepare_flag=True
                self.adjust_func()
                self.prepare_flag=False
        self.reward_list.append(np.mean(npc.as_array(self.reward.get_obj())))
        if len(self.reward_list)>self.trial_count:
            del self.reward_list[0]
    
    
    def train(self, train_loss, optimizer, episodes=None, jit_compile=True, pool_network=True, processes=None, num_store=1, processes_her=None, processes_pr=None, window_size=None, clearing_freq=None, window_size_=None, window_size_ppo=None, window_size_pr=None, random=False, save_data=True, callbacks=None, p=None):
        avg_reward=None
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
        if self.optimizer==None:
            self.optimizer=optimizer
        self.episodes=episodes
        self.jit_compile=jit_compile
        self.pool_network=pool_network
        self.processes=processes
        self.num_store=num_store
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
        self.p=p
        self.info_flag=0
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
                        self.ratio_list.append(self.initial_ratio)
                        self.TD_list.append(self.initial_TD)
                    self.prioritized_replay.ratio=None
                    self.prioritized_replay.TD=None
                else:
                    self.TD_list=manager.list()
                    for _ in range(processes):
                        self.TD_list.append(self.initial_TD)
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
                if self.stop_training==True:
                    return
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_begin'):
                        callback.on_episode_begin(i, logs={})
                train_loss.reset_states()
                if pool_network==True:
                    self.prepare(lock_list)
                    loss=self.train1(train_loss, self.optimizer)
                else:
                    loss=self.train2(train_loss,self.optimizer)
                episode_logs = {'loss': loss}
                episode_logs['reward'] = self.reward_list[-1]
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_end'):
                        callback.on_episode_end(i, logs=episode_logs)
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
                            if p!=0:
                                print('episode:{0}'.format(self.total_episode))
                                print('average reward:{0}'.format(avg_reward))
                                print()
                                print('time:{0}s'.format(self.total_time))
                            return
                if p!=0:
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
                if self.stop_training==True:
                    return
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_begin'):
                        callback.on_episode_begin(i, logs={})
                train_loss.reset_states()
                if pool_network==True:
                    self.prepare(lock_list)
                    loss=self.train1(train_loss, self.optimizer)
                else:
                    loss=self.train2(train_loss,self.optimizer)
                episode_logs = {'loss': loss}
                episode_logs['reward'] = self.reward_list[-1]
                for callback in self.callbacks:
                    if hasattr(callback, 'on_episode_end'):
                        callback.on_episode_end(i, logs=episode_logs)
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
                            if p!=0:
                                print('episode:{0}'.format(self.total_episode))
                                print('average reward:{0}'.format(avg_reward))
                                print()
                                print('time:{0}s'.format(self.total_time))
                            return
                if p!=0:
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
        if p!=0:
            print('time:{0}s'.format(self.time))
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(logs={})
        return
    
    
    def distributed_training(self, optimizer, strategy, episodes=None, num_episodes=None, jit_compile=True, pool_network=True, processes=None, num_store=1, processes_her=None, processes_pr=None, window_size=None, clearing_freq=None, window_size_=None, window_size_ppo=None, window_size_pr=None, random=False, save_data=True, callbacks=None, p=None):
        avg_reward=None
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
        if self.optimizer==None:
            self.optimizer=optimizer
        self.strategy=strategy
        self.episodes=episodes
        self.num_episodes=num_episodes
        self.jit_compile=jit_compile
        self.pool_network=pool_network
        self.processes=processes
        self.num_store=num_store
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
        self.p=p
        self.info_flag=1
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
                        self.ratio_list.append(self.initial_ratio)
                        self.TD_list.append(self.initial_TD)
                    self.prioritized_replay.ratio=None
                    self.prioritized_replay.TD=None
                else:
                    self.TD_list=manager.list()
                    for _ in range(processes):
                        self.TD_list.append(self.initial_TD)
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
                    if self.stop_training==True:
                        return
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
                    if pool_network==True:
                        self.prepare(lock_list)
                        loss=self.train1(None, self.optimizer)
                    else:
                        loss=self.train2(None,self.optimizer)
                    episode_logs = {'loss': loss}
                    episode_logs['reward'] = self.reward_list[-1]
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_end'):
                            callback.on_episode_end(i, logs=episode_logs)
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
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
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
                    if self.stop_training==True:
                        return
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
                    if pool_network==True:
                        self.prepare(lock_list)
                        loss=self.train1(None, self.optimizer)
                    else:
                        loss=self.train2(None,self.optimizer)
                    episode_logs = {'loss': loss}
                    episode_logs['reward'] = self.reward_list[-1]
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_end'):
                            callback.on_episode_end(i, logs=episode_logs)
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
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
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
        elif isinstance(strategy,tf.distribute.MultiWorkerMirroredStrategy):
            if num_episodes!=None:
                episode = 0
                self.step_in_episode = 0
                while episode < num_episodes:
                    t1=time.time()
                    if self.stop_training==True:
                        return
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
                    if pool_network==True:
                        self.prepare(lock_list)
                        loss=self.train1(None, self.optimizer)
                    else:
                        loss=self.train2(None,self.optimizer)
                        
                    if self.path!=None and episode%self.save_freq==0:
                        if self.save_param_only==False:
                            self.save_param_(self.path)
                        else:
                            self.save_(self.path)
                  
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
                            avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                            if self.criterion!=None and avg_reward>=self.criterion:
                                t2=time.time()
                                self.total_time+=(t2-t1)
                                time_=self.total_time-int(self.total_time)
                                if time_<0.5:
                                    self.total_time=int(self.total_time)
                                else:
                                    self.total_time=int(self.total_time)+1
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
                        if episode%p==0:
                            if len(self.state_pool)>=self.batch:
                                print('episode:{0}   loss:{1:.4f}'.format(episode+1,loss))
                            if avg_reward!=None:
                                print('episode:{0}   average reward:{1}'.format(episode+1,avg_reward))
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
                    if self.stop_training==True:
                        return
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
                    if pool_network==True:
                        self.prepare(lock_list)
                        loss=self.train1(None, self.optimizer)
                    else:
                        loss=self.train2(None,self.optimizer)
                        
                    if self.path!=None and episode%self.save_freq==0:
                        if self.save_param_only==False:
                            self.save_param_(self.path)
                        else:
                            self.save_(self.path)
                  
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
                            avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                            if self.criterion!=None and avg_reward>=self.criterion:
                                t2=time.time()
                                self.total_time+=(t2-t1)
                                time_=self.total_time-int(self.total_time)
                                if time_<0.5:
                                    self.total_time=int(self.total_time)
                                else:
                                    self.total_time=int(self.total_time)+1
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
                        if episode%p==0:
                            if len(self.state_pool)>=self.batch:
                                print('episode:{0}   loss:{1:.4f}'.format(episode+1,loss))
                            if avg_reward!=None:
                                print('episode:{0}   average reward:{1}'.format(episode+1,avg_reward))
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
                    if self.stop_training==True:
                        return
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_episode_begin'):
                            callback.on_episode_begin(i, logs={})
                    if pool_network==True:
                        self.prepare(lock_list)
                        loss=self.train1(None, self.optimizer)
                    else:
                        loss=self.train2(None,self.optimizer)
                        
                    if self.path!=None and episode%self.save_freq==0:
                        if self.save_param_only==False:
                            self.save_param_(self.path)
                        else:
                            self.save_(self.path)
                  
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
                            avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                            if self.criterion!=None and avg_reward>=self.criterion:
                                t2=time.time()
                                self.total_time+=(t2-t1)
                                time_=self.total_time-int(self.total_time)
                                if time_<0.5:
                                    self.total_time=int(self.total_time)
                                else:
                                    self.total_time=int(self.total_time)+1
                                if p!=0:
                                    print('episode:{0}'.format(self.total_episode))
                                    print('average reward:{0}'.format(avg_reward))
                                    print()
                                    print('time:{0}s'.format(self.total_time))
                                return
                    if p!=0:
                        if episode%p==0:
                            if len(self.state_pool)>=self.batch:
                                print('episode:{0}   loss:{1:.4f}'.format(episode+1,loss))
                            if avg_reward!=None:
                                print('episode:{0}   average reward:{1}'.format(episode+1,avg_reward))
                            else:
                                print('episode:{0}   reward:{1}'.format(episode+1,self.reward))
                            print()
                    t2=time.time()
                    self.time+=(t2-t1)
        time_=self.time-int(self.time)
        if time_<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
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
        nn.assign(self.param,param)
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
        param=self.param
        self.param=None
        pickle.dump(self,output_file)
        pickle.dump(param,output_file)
        self.param=param
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
                self.optimizer[i].built=False
                self.optimizer[i].build(self.optimizer[i]._trainable_variables)
                self.optimizer[i].load_own_variables(state_dict[i])
        else:
            state_dict=pickle.load(input_file)
            self.optimizer.built=False
            self.optimizer.build(self.optimizer._trainable_variables)
            self.optimizer.load_own_variables(state_dict)
        input_file.close()
        return