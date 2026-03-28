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

    def get_leaf_batch(self, values: np.ndarray) -> np.ndarray:
        values = values.copy().astype(np.float32)
        parent = np.zeros(len(values), dtype=np.int32)

        while True:
            left = 2 * parent + 1
            right = left + 1
            is_leaf = left >= len(self.tree)
            if np.all(is_leaf):
                break
            left_val = np.where(is_leaf, np.inf, self.tree[left])
            go_right = (~is_leaf) & (values > left_val)
            values = np.where(go_right, values - self.tree[left], values)
            parent = np.where(is_leaf, parent,
                     np.where(go_right, right, left))

        return parent - (self.capacity - 1)
    
    def update_batch(self, data_indices: np.ndarray, priorities: np.ndarray):
        tree_indices = self.capacity - 1 + data_indices
        changes = priorities - self.tree[tree_indices]
        self.tree[tree_indices] = priorities
        for idx, change in zip(tree_indices, changes):
            self._propagate(int(idx), float(change))

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
        total = self.sum_tree.total()
        segment = total / batch_size

        lo = np.arange(batch_size, dtype=np.float32) * segment
        hi = lo + segment
        vals = np.random.uniform(lo, hi).astype(np.float32)

        self.index = np.clip(
            self.sum_tree.get_leaf_batch(vals),
            0, len(train_data) - 1
        ).astype(np.int32)

        self.batch = batch_size
        return train_data[self.index], train_labels[self.index]
    
    def update_loss(self, loss=None, index=None):
        if loss is not None:
            loss=tf.cast(loss,tf.float32)
            self.loss_.assign(loss)
        elif index is not None:
            self.loss[index[0]:index[1]]=self.loss_[:self.batch]
        return

    def update(self):
        td_errors = tf.abs(self.loss_[:self.batch]).numpy()
        prios = (np.abs(td_errors) + 1e-7) ** self.alpha

        self.loss[self.index] = self.loss_[:self.batch]

        self.sum_tree.update_batch(self.index, prios)
