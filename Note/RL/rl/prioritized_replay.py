import tensorflow as tf
import torch
import numpy as np
import multiprocessing as mp


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
        self.tree = mp.Array('f', 2 * capacity - 1)
    
    def _get_buffer(self):
        return np.frombuffer(self.tree.get_obj(), dtype=np.float32)

    def _propagate(self, idx: int, change: float):
        tree = self._get_buffer()
        parent = (idx - 1) // 2
        while parent >= 0:
            tree[parent] += change
            parent = (parent - 1) // 2

    def update(self, data_idx: int, priority: float):
        tree = self._get_buffer()
        tree_idx = self.capacity - 1 + data_idx
        change = priority - tree[tree_idx]
        tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        
    def update_batch(self, data_indices: np.ndarray, priorities: np.ndarray):
        tree = self._get_buffer()
        tree_indices = self.capacity - 1 + data_indices
        changes = priorities - tree[tree_indices]
        tree[tree_indices] = priorities
        for idx, change in zip(tree_indices, changes):
            self._propagate(int(idx), float(change))

    def get_leaf(self, value: float):
        tree = self._get_buffer()
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(tree):
                leaf_idx = parent_idx
                break
            if value <= tree[left]:
                parent_idx = left
            else:
                value -= tree[left]
                parent_idx = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, tree[leaf_idx], data_idx

    def get_leaf_batch(self, values: np.ndarray) -> np.ndarray:
        tree = self._get_buffer()
        parent = np.zeros(len(values), dtype=np.int32)

        while True:
            left = 2 * parent + 1
            right = left + 1
            is_leaf = left >= len(tree)
            if np.all(is_leaf):
                break
            left_val = np.where(is_leaf, np.inf, tree[left])
            go_right = (~is_leaf) & (values > left_val)
            values = np.where(go_right, values - tree[left], values)
            parent = np.where(is_leaf, parent,
                     np.where(go_right, right, left))

        return parent - (self.capacity - 1)

    def total(self):
        tree = self._get_buffer()
        return tree[0]


class PR(pr):
    def __init__(self):
        self.sum_trees = None
        
    @tf.function(jit_compile=True)
    def compute_prios(self, td_errors, alpha):
        return tf.pow(td_errors + 1e-7, alpha)
    
    @tf.function
    def compute_prios_(self, td_errors, alpha):
        return tf.pow(td_errors + 1e-7, alpha)

    def rebuild(self, p, TD, ratio=None):
        if self.PPO:
            try:
                scores = self.lambda_ * TD + (1.0 - self.lambda_) * np.abs(ratio - 1.0)
                if self.jit_compile:
                    prios = self.compute_prios(scores, self.alpha)
                else:
                    prios = self.compute_prios_(scores, self.alpha)
            except Exception:
                scores = self.lambda_ * TD + (1.0 - self.lambda_) * torch.abs(ratio - 1.0)
                prios = torch.pow(scores + 1e-7, self.alpha)
        else:
            try:
                if self.jit_compile:
                    prios = self.compute_prios(TD, self.alpha)
                else:
                    prios = self.compute_prios_(TD, self.alpha)
            except Exception:
                prios = (TD + 1e-7) ** self.alpha

        np.frombuffer(self.sum_trees[p].tree.get_obj(), dtype=np.float32).fill(0.0)
        for i, prio in enumerate(prios):
            self.sum_trees[p].update(i, float(prio))

    def sample(self, state_pool, action_pool, next_state_pool, reward_pool, done_pool,
               lambda_, alpha, batch_size):
        self.lambda_ = lambda_

        totals = [t.total() for t in self.sum_trees]
        grand_total = sum(totals)
        segment = grand_total / batch_size

        lo = np.arange(batch_size, dtype=np.float32) * segment
        hi = lo + segment
        vals = np.random.uniform(lo, hi).astype(np.float32)  # shape: (batch_size,)

        indices = np.empty(batch_size, dtype=np.int32)
        assigned = np.zeros(batch_size, dtype=bool)
        remaining = vals.copy()
        offset = 0

        for proc, t in enumerate(self.sum_trees):
            mask = (~assigned) & (remaining <= totals[proc])
            if mask.any():
                local_idxs = t.get_leaf_batch(remaining[mask])
                global_idxs = offset + local_idxs
                indices[mask] = np.clip(global_idxs, 0, len(state_pool) - 1)
                assigned[mask] = True
            remaining[~assigned] -= totals[proc]
            offset += self.length_list[proc]

        self.index = indices
        try:
            self.batch.assign(batch_size)
        except Exception:
            self.batch = batch_size

        return (state_pool[self.index], action_pool[self.index],
                next_state_pool[self.index], reward_pool[self.index],
                done_pool[self.index])

    def update(self):
        try:
            if self.PPO:
                score = (self.lambda_ * tf.abs(self.TD_[:self.batch]) +
                         (1.0 - self.lambda_) * tf.abs(self.ratio_[:self.batch] - 1.0))
                td_errors = score.numpy()
            else:
                td_errors = tf.abs(self.TD_[:self.batch]).numpy()
        except Exception:
            if self.PPO:
                score = (self.lambda_ * torch.abs(self.TD_[:self.batch]) +
                         (1.0 - self.lambda_) * torch.abs(self.ratio_[:self.batch] - 1.0))
                td_errors = score.numpy()
            else:
                td_errors = torch.abs(self.TD_[:self.batch]).numpy()

        prios = (np.abs(td_errors) + 1e-7) ** self.alpha
        global_indices = self.index

        cumlen = 0
        for proc, length in enumerate(self.length_list):
            mask = (global_indices >= cumlen) & (global_indices < cumlen + length)
            if mask.any():
                local_idxs = (global_indices[mask] - cumlen).astype(np.int32)
                self.sum_trees[proc].update_batch(local_idxs, prios[mask])
            cumlen += length


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
