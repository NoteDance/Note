from matplotlib import pyplot as plt
import tensorflow as tf
from Note import nn
from tensorflow.python.util import nest
import numpy as np


class OptFinder:
    def __init__(self, model, optimizers):
        self.model = model
        self.optimizers = optimizers
        self.losses = []
        self.best_opt = None
        self.best_loss = 1e9

    def on_epoch_end(self, epoch, logs):
        # Log the loss
        loss = logs['loss']
        
        if epoch+1 == self.epochs:
            self.losses.append(loss)
            if loss < self.best_loss:
                self.best_opt = self.model.optimizer
                self.best_loss = loss

    def find(self, train_ds=None, loss_object=None, train_loss=None, strategy=None, batch_size=64, epochs=1, jit_compile=True):
        # Save weights into a file
        initial_weights = [tf.Variable(param.read_value()) for param in nest.flatten(self.model.param)]
        
        self.epochs = epochs

        callback = nn.LambdaCallback(on_epoch_end=lambda epoch, logs: self.on_epoch_end(epoch, logs))
        
        for opt in self.optimizers:
            self.model.optimizer = opt
            
            if strategy == None:
                self.model.train(train_ds=train_ds,
                               loss_object=loss_object, 
                               train_loss=train_loss, 
                               epochs=epochs,
                               callbacks=[callback],
                               jit_compile=jit_compile)
            else:
                self.model.distributed_training(train_dataset=train_ds,
                               loss_object=loss_object, 
                               global_batch_size=batch_size, 
                               epochs=epochs,
                               strategy=strategy,
                               callbacks=[callback],
                               jit_compile=jit_compile)
            
            nn.assign_param(self.model.param, initial_weights)

        # Restore the weights to the state before model fitting
        nn.assign_param(self.model.param, initial_weights)

    def plot_loss(self, x_scale='linear'):
        plt.ylabel("Loss")
        x_values = list(range(len(self.losses)))
        plt.xlabel("Optimizer Index")
        plt.plot(x_values, self.losses)
        plt.xscale(x_scale)
        plt.show()
    
    def plot_loss_change(self, sma=1, y_lim=(-0.01, 0.01)):
        derivatives = self.get_derivatives(sma)
        x_values = list(range(len(derivatives)))
        xlabel = "Optimizer Index"
        x_scale = 'linear'
        plt.ylabel("Rate of Loss Change")
        plt.xlabel(xlabel)
        plt.plot(x_values, derivatives)
        plt.xscale(x_scale)
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        assert sma >= 1
        n = len(self.losses)
        derivatives = [0] * sma
        for i in range(sma, n):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives


class OptFinder_rl:
    def __init__(self, agent, optimizers):
        self.agent = agent
        self.optimizers = optimizers
        self.rewards = []
        self.losses = []
        self.mean_rewards = []
        self.mean_losses = []
        self.best_reward = -1e9
        self.best_loss = 1e9
            
    def on_episode_end(self, episode, logs):
        reward = logs['reward']
        self.rewards.append(reward)
        
        if episode+1 == self.episode:
            mean_reward = np.mean(self.rewards)
            self.mean_rewards.append(mean_reward)
            if mean_reward > self.best_reward:
                self.best_opt = self.agent.optimizer
                self.best_reward = mean_reward
    
    def on_episode_end_(self, episode, logs):
        loss = logs['loss']
        self.losses.append(loss)
        
        if episode+1 == self.episode:
            mean_loss = np.mean(self.losses)
            self.mean_losses.append(mean_loss)
            if mean_loss < self.best_loss:
                self.best_opt = self.agent.optimizer
                self.best_loss = mean_loss

    def find(self, train_loss=None, pool_network=True, processes=None, processes_her=None, processes_pr=None, strategy=None, episodes=1, metrics='reward', jit_compile=True):
        # Save weights into a file
        initial_weights = [tf.Variable(param.read_value()) for param in nest.flatten(self.agent.param)]
        
        self.episodes = episodes
        
        if metrics == 'reward':
            callback = nn.LambdaCallback(on_episode_end=lambda episode, logs: self.on_episode_end(episode, logs))
        else:
            callback = nn.LambdaCallback(on_episode_end=lambda episode, logs: self.on_episode_end_(episode, logs))
        
        for opt in self.optimizers:
            self.agent.optimizer = opt
            
            if strategy == None:
                self.agent.train(train_loss=train_loss, 
                               episodes=episodes,
                               pool_network=pool_network,
                               processes=processes,
                               processes_her=processes_her,
                               processes_pr=processes_her,
                               callbacks=[callback],
                               jit_compile=jit_compile)
            else:
                self.agent.distributed_training(strategy=strategy,
                               episodes=episodes,
                               pool_network=pool_network,
                               processes=processes,
                               processes_her=processes_her,
                               processes_pr=processes_her,
                               callbacks=[callback],
                               jit_compile=jit_compile)
            
            if pool_network==True:
                for i in range(processes):
                    self.agent.state_pool_list[i] = None
                    self.agent.action_pool_list[i] = None
                    self.agent.next_state_pool_list[i] = None
                    self.agent.reward_pool_list[i] = None
                    self.agent.done_pool_list[i] = None
                if processes_her!=None or processes_pr!=None:
                    if processes_her!=None:
                        for i in range(processes_her):
                            self.agent.state_list[i] = None
                            self.agent.action_list[i] = None
                            self.agent.next_state_list[i] = None
                            self.agent.reward_list[i] = None
                            self.agent.done_list[i] = None
                    else:
                        for i in range(processes_pr):
                            self.agent.state_list[i] = None
                            self.agent.action_list[i] = None
                            self.agent.next_state_list[i] = None
                            self.agent.reward_list[i] = None
                            self.agent.done_list[i] = None
            else:
                self.agent.state_pool = None
                self.agent.action_pool = None
                self.agent.next_state_pool = None
                self.agent.reward_pool = None
                self.agent.done_pool = None
                
            if metrics == 'reward':
                self.rewards.clear()
            else:
                self.losses.clear()

            # Restore the weights to the state before agent fitting
            nn.assign_param(self.agent.param, initial_weights)

        # Restore the weights to the state before agent fitting
        nn.assign_param(self.agent.param, initial_weights)
        
    def plot_reward(self, x_scale='linear'):
        plt.ylabel("Reward")
        x_values = list(range(len(self.mean_rewards)))
        plt.xlabel("Optimizer Index")
        plt.plot(x_values, self.mean_rewards)
        plt.xscale(x_scale)
        plt.show()
    
    def plot_reward_change(self, sma=1, y_lim=(-0.01, 0.01)):
        derivatives = self.get_derivatives(sma)
        x_values = list(range(len(derivatives)))
        xlabel = "Optimizer Index"
        x_scale = 'linear'
        plt.ylabel("Rate of Reward Change")
        plt.xlabel(xlabel)
        plt.plot(x_values, derivatives)
        plt.xscale(x_scale)
        plt.ylim(y_lim)
        plt.show()
    
    def get_derivatives(self, sma):
        assert sma >= 1
        n = len(self.mean_rewards)
        derivatives = [0] * sma
        for i in range(sma, n):
            derivatives.append((self.mean_rewards[i] - self.mean_rewards[i - sma]) / sma)
        return derivatives
    
    def plot_loss(self, x_scale='linear'):
        plt.ylabel("Loss")
        x_values = list(range(len(self.mean_losses)))
        plt.xlabel("Optimizer Index")
        plt.plot(x_values, self.mean_losses)
        plt.xscale(x_scale)
        plt.show()
    
    def plot_loss_change(self, sma=1, y_lim=(-0.01, 0.01)):
        derivatives = self.get_derivatives_(sma)
        x_values = list(range(len(derivatives)))
        xlabel = "Optimizer Index"
        x_scale = 'linear'
        plt.ylabel("Rate of Loss Change")
        plt.xlabel(xlabel)
        plt.plot(x_values, derivatives)
        plt.xscale(x_scale)
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives_(self, sma):
        assert sma >= 1
        n = len(self.mean_losses)
        derivatives = [0] * sma
        for i in range(sma, n):
            derivatives.append((self.mean_losses[i] - self.mean_losses[i - sma]) / sma)
        return derivatives
