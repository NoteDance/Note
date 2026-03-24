import tensorflow as tf
import torch
import numpy as np


class pr:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.index=None
        self.PPO=False
        self.jit_compile=True
    
    
    @tf.function(jit_compile=True)
    def compute_probs(self, td_errors, alpha):
        prios = tf.pow(td_errors + 1e-7, alpha)
        return prios / tf.reduce_sum(prios)
    
    
    @tf.function
    def compute_probs_(self, td_errors, alpha):
        prios = tf.pow(td_errors + 1e-7, alpha)
        return prios / tf.reduce_sum(prios)
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,lambda_,alpha,batch):
        if self.PPO:
            try:
                scores=self.lambda_*self.TD+(1.0-self.lambda_)*tf.abs(self.ratio-1.0)
                if self.jit_compile:
                    p=self.compute_probs(scores, alpha)
                else:
                    p=self.compute_probs_(scores, alpha)
            except Exception:
                scores=self.lambda_*self.TD+(1.0-self.lambda_)*torch.abs(self.ratio-1.0)
                prios=torch.pow(scores+1e-7,alpha)
                p=prios/torch.sum(prios)
        else:
            try:
                if self.jit_compile:
                    p=self.compute_probs(self.TD, alpha)
                else:
                    p=self.compute_probs_(self.TD, alpha)
            except Exception:
                prios=(self.TD+1e-7)**alpha
                p=prios/torch.sum(prios)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p.numpy(),replace=False)
        try:
            self.batch.assign(batch)
        except Exception:
            self.batch=batch
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update(self,TD=None,ratio=None):
        if self.PPO:
            if TD is not None:
                TD=tf.cast(TD,tf.float32)
                ratio=tf.cast(ratio,tf.float32)
                self.TD_[:self.batch].assign(TD)
                self.ratio_[:self.batch].assign(ratio)
            else:
                self.ratio[self.index]=self.ratio_[:self.batch]
                try:
                    self.TD[self.index]=tf.abs(self.TD_[:self.batch])
                except Exception:
                    self.TD[self.index]=torch.abs(self.TD_[:self.batch])
        else:
            if TD is not None:
                TD=tf.cast(TD,tf.float32)
                self.TD_[:self.batch].assign(TD)
            else:
                try:
                    self.TD[self.index]=tf.abs(self.TD_[:self.batch])
                except Exception:
                    self.TD[self.index]=torch.abs(self.TD_[:self.batch])
        return


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        self.full = False

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float):
        tree_idx = self.capacity - 1 + self.data_pointer
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.data_pointer == 0:
            self.full = True

    def get_leaf(self, value: float):
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            if value <= self.tree[left]:
                parent_idx = left
            else:
                value -= self.tree[left]
                parent_idx = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total(self):
        return self.tree[0]


class PR(pr):
    def __init__(self):
        self.sum_tree = None

    def build(self, capacity: int, alpha: float = 0.7):
        self.capacity = capacity
        self.alpha = alpha
        self.sum_tree = SumTree(capacity)
        
    @tf.function(jit_compile=True)
    def compute_prios(self, td_errors, alpha):
        return tf.pow(td_errors + 1e-7, alpha)
    
    @tf.function
    def compute_prios_(self, td_errors, alpha):
        return tf.pow(td_errors + 1e-7, alpha)

    def rebuild(self):
        if self.PPO:
            try:
                scores = self.lambda_ * self.TD + (1.0 - self.lambda_) * np.abs(self.ratio - 1.0)
                if self.jit_compile:
                    prios=self.compute_prios(scores, self.alpha)
                else:
                    prios=self.compute_prios_(scores, self.alpha)
            except Exception:
                scores=self.lambda_*self.TD+(1.0-self.lambda_)*torch.abs(self.ratio-1.0)
                prios=torch.pow(scores+1e-7,self.alpha)
        else:
            try:
                if self.jit_compile:
                    prios=self.compute_prios(self.TD, self.alpha)
                else:
                    prios=self.compute_prios_(self.TD, self.alpha)
            except Exception:
                prios=(self.TD+1e-7)**self.alpha

        self.sum_tree.tree.fill(0.0)
        for i in range(len(prios)):
            self.sum_tree.update(self.capacity - 1 + i, prios[i])

    def sample(self, state_pool, action_pool, next_state_pool, reward_pool, done_pool,
               lambda_, alpha, batch_size):
        self.lambda_ = lambda_

        indices = []
        segment = self.sum_tree.total() / batch_size
        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            _, _, data_idx = self.sum_tree.get_leaf(val)
            indices.append(data_idx)

        self.index = np.array(indices, dtype=np.int32)

        return (state_pool[indices], action_pool[indices],
                next_state_pool[indices], reward_pool[indices],
                done_pool[indices])

    def update(self, td_errors):
        for i, td in enumerate(td_errors):
            data_idx = self.index[i]
            prio = (abs(td) + 1e-7) ** self.alpha
            self.sum_tree.update(self.capacity - 1 + data_idx, prio)


class pr_mp:
    def __init__(self):
        self.TD=None
        self.lambda_=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,alpha,batch,p):
        prios=(self.TD[p]+1e-7)**alpha
        prob=prios/np.sum(prios)
        self.index[p]=np.random.choice(np.arange(len(state_pool)),size=[batch],p=prob,replace=False)
        self.batch=batch
        return state_pool[self.index[p]],action_pool[self.index[p]],next_state_pool[self.index[p]],reward_pool[self.index[p]],done_pool[self.index[p]]
    
    
    def update(self,TD=None,ratio=None,p=None):
        if TD is not None:
            try:
                TD=tf.cast(TD,tf.float32)
                self.TD_[:self.batch].assign(TD)
            except Exception:
                self.TD_[:self.batch]=TD
        else:
            self.TD[p][self.index[p]]=np.abs(self.TD_)
        return
