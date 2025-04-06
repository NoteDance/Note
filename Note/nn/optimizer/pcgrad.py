""" PCGrad
https://arxiv.org/abs/2001.06782

Copyright 2025 NoteDance
"""
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import random


def flatten_grad(grads):
    r"""Flatten the gradient."""
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)


def un_flatten_grad(grads, shapes):
    r"""Unflatten the gradient."""
    idx = 0
    un_flatten_grads = []
    for shape in shapes:
        length = np.prod(shape)
        un_flatten_grads.append(tf.reshape(grads[idx:idx + length], shape))
        idx += length
    return un_flatten_grads


class PCGrad:
    r"""Gradient Surgery for Multi-Task Learning.

    :param reduction: str. reduction method.
    """
    def __init__(self, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError("Reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def pack_grad(self, tape, losses, variables):
        """
        Compute gradients for each loss and flatten them.
        
        Parameters:
          tape: A tf.GradientTape instance (should be persistent if used for multiple losses).
          losses: A list of loss tensors corresponding to each task.
          variables: List of model variables.
          
        Returns:
          grads_list: A list of flattened gradients for each task.
          shapes: A list of shapes for each variable.
          has_grads_list: A list of flattened masks (1 if the gradient exists, 0 otherwise) for each task.
        """
        grads_list = []
        shapes = [v.shape for v in variables]
        has_grads_list = []
        for loss in losses:
            grads = tape.gradient(loss, variables)
            grads_list_ = []
            has_grads_list_ = []
            for g, v in zip(grads, variables):
                if g is None:
                    g = tf.zeros_like(v)
                    has_val = tf.zeros_like(v)
                else:
                    has_val = tf.ones_like(v)
                grads_list_.append(tf.reshape(g, [-1]))
                has_grads_list_.append(has_val)
            grads_list.append(flatten_grad(grads_list_))
            has_grads_list.append(flatten_grad(has_grads_list_))
        return grads_list, shapes, has_grads_list

    def project_conflicting(self, grads, has_grads):
        """
        Project conflicting gradients. For each task's gradient, randomly iterate
        over other tasks' gradients and subtract the projection if the dot product is negative.
        
        Parameters:
          grads: A list of flattened gradients for each task.
          has_grads: A list of flattened masks indicating gradient existence.
          
        Returns:
          merged_grad: The merged flattened gradient after conflict resolution.
        """
        shared = tf.cast(
            tf.reduce_prod(
                tf.stack([tf.cast(h, tf.int32) for h in has_grads]),
                axis=0),
            tf.bool)
        pc_grad = [g for g in grads]
        for i in range(len(pc_grad)):
            g_i = pc_grad[i]
            random.shuffle(grads)
            for g_j in grads:
                g_i_flat = tf.reshape(g_i, [-1])
                g_j_flat = tf.reshape(g_j, [-1])
                dot = tf.tensordot(g_i_flat, g_j_flat, axes=1)
                if dot < 0:
                    norm_sq = tf.reduce_sum(tf.square(g_j_flat))
                    proj = dot * g_j / norm_sq
                    pc_grad[i] = pc_grad[i] - proj
        stacked_pc_grad = tf.stack(pc_grad)
        mask = tf.cast(shared, stacked_pc_grad.dtype)
        shared_grads = stacked_pc_grad * mask
        non_shared_grads   = stacked_pc_grad * (1. - mask)
        if self.reduction == 'mean':
            merged_shared_grads = tf.reduce_mean(shared_grads, axis=0)
        else:
            merged_shared_grads = tf.reduce_sum(shared_grads, axis=0)
        merged_non_shared_grads = tf.reduce_sum(non_shared_grads, axis=0)
        return merged_shared_grads + merged_non_shared_grads

    def pc_backward(self, tape, losses, variables):
        """
        Compute the gradients for multiple losses using PCGrad and apply them to update parameters.
        
        Parameters:
          tape: A tf.GradientTape instance (should be persistent if used for multiple losses).
          losses: A list of loss tensors for each task.
          variables: List of model variables.
        """
        grads, shapes, has_grads = self.pack_grad(tape, losses, variables)
        pc_grad = self.project_conflicting(grads, has_grads)
        pc_grad = un_flatten_grad(pc_grad, shapes)
        return pc_grad


class PPCGrad:
    r"""Gradient Surgery for Multi-Task Learning.

    :param reduction: str. reduction method.
    """
    def __init__(self, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError("Reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.manager = mp.Manager()

    def pack_grad(self, tape, losses, variables):
        """
        Compute gradients for each loss and flatten them.
        
        Parameters:
          tape: A tf.GradientTape instance (should be persistent if used for multiple losses).
          losses: A list of loss tensors corresponding to each task.
          variables: List of model variables.
          
        Returns:
          grads_list: A list of flattened gradients for each task.
          shapes: A list of shapes for each variable.
          has_grads_list: A list of flattened masks (1 if the gradient exists, 0 otherwise) for each task.
        """
        grads_list = []
        shapes = [v.shape for v in variables]
        has_grads_list = []
        for loss in losses:
            grads = tape.gradient(loss, variables)
            grads_list_ = []
            has_grads_list_ = []
            for g, v in zip(grads, variables):
                if g is None:
                    g = tf.zeros_like(v)
                    has_val = tf.zeros_like(v)
                else:
                    has_val = tf.ones_like(v)
                grads_list_.append(tf.reshape(g, [-1]))
                has_grads_list_.append(has_val)
            grads_list.append(flatten_grad(grads_list_))
            has_grads_list.append(flatten_grad(has_grads_list_))
        return grads_list, shapes, has_grads_list

    def project_conflicting(self, grads, has_grads):
        """
        Project conflicting gradients. For each task's gradient, randomly iterate
        over other tasks' gradients and subtract the projection if the dot product is negative.
        
        Parameters:
          grads: A list of flattened gradients for each task.
          has_grads: A list of flattened masks indicating gradient existence.
          
        Returns:
          merged_grad: The merged flattened gradient after conflict resolution.
        """
        shared = tf.cast(
            tf.reduce_prod(
                tf.stack([tf.cast(h, tf.int32) for h in has_grads]),
                axis=0),
            tf.bool)

        def project_conflicting_gradient(g_i, grads):
            random.shuffle(grads)
            for g_j in grads:
                g_i_flat = tf.reshape(g_i, [-1])
                g_j_flat = tf.reshape(g_j, [-1])
                dot = tf.tensordot(g_i_flat, g_j_flat, axes=1)
                if dot < 0:
                    norm_sq = tf.reduce_sum(tf.square(g_j_flat))
                    proj = dot * g_j / norm_sq
                    pc_grad[i] = pc_grad[i] - proj
                    
        pc_grad = self.manager.list([g for g in grads])
        grads = self.manager.list(grads)
        process_list=[]
        for i in range(len(pc_grad)):
            g_i = pc_grad[i]
            process=mp.Process(target=project_conflicting_gradient,args=(g_i,grads))
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()
        stacked_pc_grad = tf.stack(pc_grad)
        mask = tf.cast(shared, stacked_pc_grad.dtype)
        shared_grads = stacked_pc_grad * mask
        non_shared_grads   = stacked_pc_grad * (1. - mask)
        if self.reduction == 'mean':
            merged_shared_grads = tf.reduce_mean(shared_grads, axis=0)
        else:
            merged_shared_grads = tf.reduce_sum(shared_grads, axis=0)
        merged_non_shared_grads = tf.reduce_sum(non_shared_grads, axis=0)
        return merged_shared_grads + merged_non_shared_grads

    def pc_backward(self, tape, losses, variables):
        """
        Compute the gradients for multiple losses using PCGrad and apply them to update parameters.
        
        Parameters:
          tape: A tf.GradientTape instance (should be persistent if used for multiple losses).
          losses: A list of loss tensors for each task.
          variables: List of model variables.
        """
        grads, shapes, has_grads = self.pack_grad(tape, losses, variables)
        pc_grad = self.project_conflicting(grads, has_grads)
        pc_grad = un_flatten_grad(pc_grad, shapes)
        return pc_grad