import tensorflow as tf
import numpy as np


class SumTree:
    def __init__(self, capacity: int, pool_network: bool):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
    
    def _get_buffer(self):
        return np.frombuffer(self.tree.get_obj(), dtype=np.float32)

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def update(self, data_idx: int, priority: float):
        tree_idx = self.capacity - 1 + data_idx
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

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


class PR:
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

    def sample(self, train_data, train_labels, alpha, batch_size):
        indices = []
        segment = self.sum_tree.total() / batch_size
        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            _, _, data_idx = self.sum_tree.get_leaf(val)
            indices.append(data_idx)

        self.index = np.array(indices, dtype=np.int32)
        
        self.batch=batch_size

        return train_data[indices], train_labels[indices]
    
    def update_loss(self, loss=None, index=None):
        if loss is not None:
            loss=tf.cast(loss,tf.float32)
            self.loss_.assign(loss)
        elif index is not None:
            self.loss[index[0]:index[1]]=self.loss_[:self.batch]
        return

    def update(self):
        td_errors = tf.abs(self.loss_[:self.batch]).numpy()
        for i, td in enumerate(td_errors):
            data_idx = self.index[i]
            prio = (abs(td) + 1e-7) ** self.alpha
            self.sum_tree.update(data_idx, prio)
        self.loss[self.index]=self.loss_[:self.batch]
