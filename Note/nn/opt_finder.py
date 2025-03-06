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
        self.losses.append(loss)

        if loss < self.best_loss:
            self.best_opt = self.model.optimizer
            self.best_loss = loss

    def find(self, train_ds=None, loss_object=None, train_loss=None, strategy=None, batch_size=64, jit_compile=True):
        # Save weights into a file
        initial_weights = [tf.Variable(param.read_value()) for param in nest.flatten(self.model.param)]

        callback = nn.LambdaCallback(on_epoch_end=lambda epoch, logs: self.on_epoch_end(epoch, logs))
        
        for opt in self.optimizers:
            self.model.optimizer = opt
            
            if strategy == None:
                self.model.train(train_ds=train_ds,
                               loss_object=loss_object, 
                               train_loss=train_loss, 
                               epochs=1,
                               callbacks=[callback],
                               jit_compile=jit_compile)
            else:
                self.model.distributed_training(train_dataset=train_ds,
                               loss_object=loss_object, 
                               global_batch_size=batch_size, 
                               epochs=1,
                               strategy=strategy,
                               callbacks=[callback],
                               jit_compile=jit_compile)
            
            nn.assign_param(self.model.param, initial_weights)

        # Restore the weights to the state before model fitting
        nn.assign_param(self.model.param, initial_weights)

    def plot_loss(self, n_skip_beginning=0, n_skip_end=0, x_scale='linear'):
        plt.ylabel("Loss")
        x_values = list(range(len(self.losses)))[n_skip_beginning: -n_skip_end]
        plt.xlabel("Optimizer Index")
        plt.plot(x_values, self.losses[n_skip_beginning: -n_skip_end])
        plt.xscale(x_scale)
        plt.show()
    
    def plot_loss_change(self, sma=1, n_skip_beginning=0, n_skip_end=0, y_lim=(-0.01, 0.01)):
        derivatives = self.get_derivatives(sma)[n_skip_beginning: -n_skip_end]
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
        self.normalized_reward = []
        self.best_reward = -1e9
            
    def on_episode_end(self, episode, logs):
        reward = logs['reward']
        self.rewards.append(reward)
        
        recent_rewards = self.rewards
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards) + 1e-8
        normalized_reward = (reward - mean_reward) / std_reward
        self.normalized_reward.append(normalized_reward)
        if normalized_reward > self.best_reward:
            self.best_opt = self.model.optimizer
            self.best_reward = normalized_reward

    def find(self, train_loss=None, pool_network=True, processes=None, processes_her=None, processes_pr=None, strategy=None, episodes=1, jit_compile=True):
        # Save weights into a file
        initial_weights = [tf.Variable(param.read_value()) for param in nest.flatten(self.agent.param)]

        callback = nn.LambdaCallback(on_episode_end=lambda episode, logs: self.on_episode_end(episode, logs))
        
        for opt in self.optimizers:
            self.model.optimizer = opt
            
            if strategy == None:
                self.model.train(train_loss=train_loss, 
                               episodes=episodes,
                               pool_network=pool_network,
                               processes=processes,
                               processes_her=processes_her,
                               processes_pr=processes_her,
                               callbacks=[callback],
                               jit_compile=jit_compile)
            else:
                self.model.distributed_training(strategy=strategy,
                               episodes=episodes,
                               pool_network=pool_network,
                               processes=processes,
                               processes_her=processes_her,
                               processes_pr=processes_her,
                               callbacks=[callback],
                               jit_compile=jit_compile)

            # Restore the weights to the state before model fitting
            nn.assign_param(self.agent.param, initial_weights)

        # Restore the weights to the state before model fitting
        nn.assign_param(self.agent.param, initial_weights)
        
    def plot_reward(self, x_scale='linear'):
        plt.ylabel("Reward")
        x_values = list(range(len(self.normalized_reward)))
        plt.xlabel("Optimizer Index")
        plt.plot(x_values, self.normalized_reward)
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
        n = len(self.normalized_reward)
        derivatives = [0] * sma
        for i in range(sma, n):
            derivatives.append((self.normalized_reward[i] - self.normalized_reward[i - sma]) / sma)
        return derivatives