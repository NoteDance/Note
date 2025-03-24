from Note import nn
import multiprocessing
import numpy as np
from functools import partial


def episode_end_callback(epoch, logs, model, lock, callback_func):
    callback_func(epoch, logs, model, lock)
    

class AgentFinder:
    def __init__(self, agents, optimizers):
        self.agents = agents
        self.optimizers = optimizers
        manager = multiprocessing.Manager()
        self.rewards = manager.dict()
        self.losses = manager.dict()
        self.logs = manager.dict()
        self.logs['best_reward'] = -1e9
        self.logs['best_loss'] = 1e9
        self.lock = multiprocessing.Lock()
            
    def on_episode_end(self, episode, logs, agent=None, lock=None):
        lock.acquire()
        reward = logs['reward']
        self.rewards[agent].append(reward)
        
        if episode+1 == self.episode:
            mean_reward = np.mean(self.rewards[agent])
            if mean_reward > self.logs['best_reward']:
                self.logs['best_opt'] = agent.optimizer
                self.logs['best_reward'] = mean_reward
        lock.release()
    
    def on_episode_end_(self, episode, logs, agent=None, lock=None):
        lock.acquire()
        loss = logs['loss']
        self.losses[agent].append(loss)
        
        if episode+1 == self.episode:
            mean_loss = np.mean(self.losses[agent])
            if mean_loss < self.logs['best_loss']:
                self.logs['best_opt'] = agent.optimizer
                self.logs['best_loss'] = mean_loss
        lock.release()

    def find(self, train_loss=None, pool_network=True, processes=None, processes_her=None, processes_pr=None, strategy=None, episodes=1, metrics='reward', jit_compile=True):
        self.episodes = episodes
        
        process_list=[]
        for i in range(len(self.agents)):
            if metrics == 'reward':
                partial_callback = partial(
                    episode_end_callback,
                    model=self.agents[i],
                    lock=self.lock,
                    callback_func=self.on_episode_end
                )
                callback = nn.LambdaCallback(on_episode_end=partial_callback)
                self.rewards[self.agents[i]] = []
            else:
                partial_callback = partial(
                    episode_end_callback,
                    model=self.agents[i],
                    lock=self.lock,
                    callback_func=self.on_episode_end_
                )
                callback = nn.LambdaCallback(on_episode_end=partial_callback)
                self.losses[self.agents[i]] = []
            self.agents[i].optimizer = self.optimizers[i]
            if strategy == None:
                process=multiprocessing.Process(target=self.agents[i].train,kwargs={
                                                        'train_loss': train_loss,
                                                        'episodes': episodes,
                                                        'pool_network': pool_network,
                                                        'processes': processes,
                                                        'processes_her': processes_her,
                                                        'processes_pr': processes_pr,
                                                        'callbacks': [callback],
                                                        'jit_compile': jit_compile
                                                    })
                process.start()
                process_list.append(process)
            else:
                if metrics == 'reward':
                    partial_callback = partial(
                        episode_end_callback,
                        model=self.agents[i],
                        lock=self.lock,
                        callback_func=self.on_episode_end
                    )
                    callback = nn.LambdaCallback(on_episode_end=partial_callback)
                    self.rewards[self.agents[i]] = []
                else:
                    partial_callback = partial(
                        episode_end_callback,
                        model=self.agents[i],
                        lock=self.lock,
                        callback_func=self.on_episode_end_
                    )
                    callback = nn.LambdaCallback(on_episode_end=partial_callback)
                self.losses[self.agents[i]] = []
                self.agents[i].optimizer = self.optimizers[i]
                process=multiprocessing.Process(target=self.agents[i].distributed_training,kwargs={
                                                        'strategy': strategy,
                                                        'episodes': episodes,
                                                        'pool_network': pool_network,
                                                        'processes': processes,
                                                        'processes_her': processes_her,
                                                        'processes_pr': processes_pr,
                                                        'callbacks': [callback],
                                                        'jit_compile': jit_compile
                                                    })
                process.start()
                process_list.append(process)
        for process in process_list:
            process.join()