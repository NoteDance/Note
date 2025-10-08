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
        self.last_svd_step: int = -1
    
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

    def get_low_rank_grad_std(self, grad):
        if grad.shape[0] >= grad.shape[1]:
            return tf.matmul(grad, tf.transpose(self.ortho_matrix))
        return tf.matmul(tf.transpose(self.ortho_matrix), grad)

    def get_low_rank_grad_reverse_std(self, grad):
        if grad.shape[0] >= grad.shape[1]:
            return tf.matmul(tf.transpose(self.ortho_matrix), grad)
        return tf.matmul(grad, tf.transpose(self.ortho_matrix))

    def get_low_rank_grad_right(self, grad):
        return tf.matmul(grad, tf.transpose(self.ortho_matrix))

    def get_low_rank_grad_left(self, grad):
        return tf.matmul(tf.transpose(self.ortho_matrix), grad)

    def get_low_rank_grad_full(self, grad):
        a, b = self.ortho_matrix
        return tf.matmul(tf.matmul(tf.transpose(a), grad), tf.transpose(b))

    def get_low_rank_grad_random(self, grad):
        is_right = grad.shape[0] >= grad.shape[1]
        if is_right:
            return tf.matmul(grad, tf.transpose(self.ortho_matrix))
        else:
            return tf.matmul(tf.transpose(self.ortho_matrix), grad)
    
    def update_ortho_matrix(self, x, from_random_matrix: bool) -> None:
        is_right: bool = x.shape[0] >= x.shape[1]

        if self.projection_type == 'std':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='right' if is_right else 'left', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'reverse_std':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='left' if is_right else 'right', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'right':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='right', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'left':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='left', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'full':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='full', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'random':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type='right' if is_right else 'left',
                from_random_matrix=from_random_matrix,
            )
        else:
            raise NotImplementedError(f'unsupported projection_type: {self.projection_type}')

    def project(
        self,
        grad,
        num_steps: int,
        svd_basis_matrix = None,
        from_random_matrix: bool = False,
    ):
        update_ortho_matrix: bool = self.ortho_matrix is None or num_steps % self.update_proj_gap == 0
        already_updated_this_step: bool = num_steps == self.last_svd_step

        if update_ortho_matrix and not already_updated_this_step:
            self.update_ortho_matrix(
                x=grad if svd_basis_matrix is None else svd_basis_matrix,
                from_random_matrix=from_random_matrix,
            )
            self.last_svd_step = num_steps

        if self.projection_type == 'std':
            return self.get_low_rank_grad_std(grad)
        if self.projection_type == 'reverse_std':
            return self.get_low_rank_grad_reverse_std(grad)
        if self.projection_type == 'right':
            return self.get_low_rank_grad_right(grad)
        if self.projection_type == 'left':
            return self.get_low_rank_grad_left(grad)
        if self.projection_type == 'full':
            return self.get_low_rank_grad_full(grad)
        if self.projection_type == 'random':
            return self.get_low_rank_grad_random(grad)

        raise NotImplementedError

    def project_back(self, low_rank_grad):
        if self.projection_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                return tf.matmul(low_rank_grad, self.ortho_matrix) * self.scale
            else:
                return tf.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        elif self.projection_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                return tf.matmul(self.ortho_matrix, tf.transpose(low_rank_grad)) * self.scale
            else:
                return tf.matmul(low_rank_grad, tf.transpose(self.ortho_matrix)) * self.scale
        elif self.projection_type == 'right':
            return tf.matmul(low_rank_grad, tf.transpose(self.ortho_matrix)) * self.scale
        elif self.projection_type == 'left':
            return tf.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        elif self.projection_type == 'full':
            a, b = self.ortho_matrix
            return tf.matmul(tf.matmul(a, low_rank_grad), tf.transpose(b)) * self.scale
        elif self.projection_type == 'random':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                return tf.matmul(low_rank_grad, tf.transpose(self.ortho_matrix)) * self.scale
            else:
                return tf.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        else:
            raise NotImplementedError
