import math
from typing import Literal, Optional

import tensorflow as tf

PROJECTION_TYPE = Literal['std', 'reverse_std', 'right', 'left', 'full', 'random']


class GaLoreProjector:
    def __init__(
        self,
        rank: Optional[int] = 128,
        update_proj_gap: int = 50,
        scale: float = 1.0,
        projection_type: PROJECTION_TYPE = 'std',
        **kwargs,
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type

        self.ortho_matrix = None
        self.last_svd_step: int = tf.Variable(-1, dtype=tf.int64)
    
    @staticmethod
    def get_orthogonal_matrix(
        weights, rank: Optional[int], projection_type: str, from_random_matrix: bool = False
    ):
        if projection_type not in {'right', 'left', 'full'}:
            raise ValueError('`projection_type` should be one of left, right or full')
        
        original_type = weights.dtype
        is_float: bool = original_type == tf.float32

        if not from_random_matrix:
            u, s, v = tf.linalg.svd(weights if is_float else tf.cast(weights, dtype=tf.float32), full_matrices=False)
            vh = tf.transpose(v)
        elif isinstance(rank, int):
            m, n = weights.shape[0], weights.shape[1]
            u = tf.random.normal((m, rank), dtype=original_type) / math.sqrt(rank)
            vh = tf.random.normal((rank, n), dtype=original_type) / math.sqrt(rank)
        else:
            raise TypeError('`rank` should be int when `from_random_matrix` is True')

        if projection_type == 'right':
            b = vh[:rank, :] if isinstance(rank, int) else vh
            return b if is_float else tf.cast(b, dtype=original_type)
        if projection_type == 'left':
            a = u[:, :rank] if isinstance(rank, int) else u
            return a if is_float else tf.cast(a, dtype=original_type)

        a = u[:, :rank] if isinstance(rank, int) else u
        b = vh[:rank, :] if isinstance(rank, int) else vh
        return ((a, b) if is_float else (tf.cast(a, dtype=original_type), tf.cast(b, dtype=original_type)))

    def get_low_rank_grad_std(self, grad, ortho_matrix):
        if ortho_matrix is not None:
            ortho_matrix = ortho_matrix
        else:
            ortho_matrix = self.ortho_matrix
        if grad.shape[0] >= grad.shape[1]:
            return tf.matmul(grad, tf.transpose(ortho_matrix))
        return tf.matmul(tf.transpose(ortho_matrix), grad)

    def get_low_rank_grad_reverse_std(self, grad, ortho_matrix):
        if ortho_matrix is not None:
            ortho_matrix = ortho_matrix
        else:
            ortho_matrix = self.ortho_matrix
        if grad.shape[0] >= grad.shape[1]:
            return tf.matmul(tf.transpose(ortho_matrix), grad)
        return tf.matmul(grad, tf.transpose(ortho_matrix))

    def get_low_rank_grad_right(self, grad, ortho_matrix):
        if ortho_matrix is not None:
            ortho_matrix = ortho_matrix
        else:
            ortho_matrix = self.ortho_matrix
        return tf.matmul(grad, tf.transpose(ortho_matrix))

    def get_low_rank_grad_left(self, grad, ortho_matrix):
        if ortho_matrix is not None:
            ortho_matrix = ortho_matrix
        else:
            ortho_matrix = self.ortho_matrix
        return tf.matmul(tf.transpose(ortho_matrix), grad)

    def get_low_rank_grad_full(self, grad, ortho_matrix):
        if ortho_matrix is not None:
            ortho_matrix = ortho_matrix
        else:
            ortho_matrix = self.ortho_matrix
        a, b = ortho_matrix
        return tf.matmul(tf.matmul(tf.transpose(a), grad), tf.transpose(b))

    def get_low_rank_grad_random(self, grad, ortho_matrix):
        if ortho_matrix is not None:
            ortho_matrix = ortho_matrix
        else:
            ortho_matrix = self.ortho_matrix
        is_right = grad.shape[0] >= grad.shape[1]
        if is_right:
            return tf.matmul(grad, tf.transpose(ortho_matrix))
        else:
            return tf.matmul(tf.transpose(ortho_matrix), grad)
    
    def update_ortho_matrix(self, x, from_random_matrix: bool) -> None:
        is_right: bool = x.shape[0] >= x.shape[1]

        if self.projection_type == 'std':
            new_ortho = self.get_orthogonal_matrix(
                x, self.rank, projection_type='right' if is_right else 'left', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'reverse_std':
            new_ortho = self.get_orthogonal_matrix(
                x, self.rank, projection_type='left' if is_right else 'right', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'right':
            new_ortho = self.get_orthogonal_matrix(
                x, self.rank, projection_type='right', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'left':
            new_ortho = self.get_orthogonal_matrix(
                x, self.rank, projection_type='left', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'full':
            a, b = self.get_orthogonal_matrix(
                x, self.rank, projection_type='full', from_random_matrix=from_random_matrix
            )
            return a, b
        elif self.projection_type == 'random':
            new_ortho = self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type='right' if is_right else 'left',
                from_random_matrix=from_random_matrix,
            )
        else:
            raise NotImplementedError(f'unsupported projection_type: {self.projection_type}')
        
        return new_ortho

    def project(
        self,
        grad,
        num_steps: int,
        svd_basis_matrix = None,
        from_random_matrix: bool = False,
    ):
        last_svd_step = tf.cast(self.last_svd_step, num_steps.dtype)
        pred = tf.logical_and(num_steps % self.update_proj_gap == 0, num_steps != last_svd_step)
        
        def true_fn():
            new_ortho = self.update_ortho_matrix(
                x=grad if svd_basis_matrix is None else svd_basis_matrix,
                from_random_matrix=from_random_matrix,
            )
            self.last_svd_step.assign(tf.cast(num_steps, self.last_svd_step.dtype))
            return new_ortho
        
        def false_fn():
            return self.ortho_matrix
        
        new_ortho = tf.cond(pred, true_fn, false_fn)
        
        if self.projection_type != 'full':
            self.ortho_matrix.assign(new_ortho)
        else:
            a, b = new_ortho
            self.ortho_matrix[0].assign(a)
            self.ortho_matrix[1].assign(b)

        if self.projection_type == 'std':
            return self.get_low_rank_grad_std(grad, None)
        if self.projection_type == 'reverse_std':
            return self.get_low_rank_grad_reverse_std(grad, None)
        if self.projection_type == 'right':
            return self.get_low_rank_grad_right(grad, None)
        if self.projection_type == 'left':
            return self.get_low_rank_grad_left(grad, None)
        if self.projection_type == 'full':
            return self.get_low_rank_grad_full(grad, None)
        if self.projection_type == 'random':
            return self.get_low_rank_grad_random(grad, None)

        raise NotImplementedError
    
    def project_(
        self,
        grad,
        ortho_matrix = None,
    ):
        if self.projection_type == 'std':
            return self.get_low_rank_grad_std(grad, ortho_matrix)
        if self.projection_type == 'reverse_std':
            return self.get_low_rank_grad_reverse_std(grad, ortho_matrix)
        if self.projection_type == 'right':
            return self.get_low_rank_grad_right(grad, ortho_matrix)
        if self.projection_type == 'left':
            return self.get_low_rank_grad_left(grad, ortho_matrix)
        if self.projection_type == 'full':
            return self.get_low_rank_grad_full(grad, ortho_matrix)
        if self.projection_type == 'random':
            return self.get_low_rank_grad_random(grad, ortho_matrix)

        raise NotImplementedError
        
    def project_back(self, low_rank_grad):
        if self.projection_type == 'std':
            return (
                tf.matmul(low_rank_grad, self.ortho_matrix)
                if low_rank_grad.shape[0] >= low_rank_grad.shape[1]
                else tf.matmul(self.ortho_matrix, low_rank_grad)
            ) * self.scale
        if self.projection_type == 'reverse_std':
            return (
                tf.matmul(self.ortho_matrix, low_rank_grad)
                if low_rank_grad.shape[0] > low_rank_grad.shape[1]
                else tf.matmul(low_rank_grad, self.ortho_matrix)
            ) * self.scale
        if self.projection_type == 'right':
            return tf.matmul(low_rank_grad, self.ortho_matrix) * self.scale
        if self.projection_type == 'left':
            return tf.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        if self.projection_type == 'full':
            return tf.matmul(tf.matmul(self.ortho_matrix[0], low_rank_grad), self.ortho_matrix[1]) * self.scale
        if self.projection_type == 'random':
            return (
                tf.matmul(low_rank_grad, self.ortho_matrix)
                if low_rank_grad.shape[0] >= low_rank_grad.shape[1]
                else tf.matmul(self.ortho_matrix, low_rank_grad)
            ) * self.scale

        raise NotImplementedError