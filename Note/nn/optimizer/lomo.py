""" LOMO
https://arxiv.org/abs/2306.09782

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from Note import nn
import os


class LOMO(optimizer.Optimizer):
    def __init__(self, model, lr=1e-3, clip_grad_norm=None, clip_grad_value=None, zero3_enabled=True, name="lomo"):
        super().__init__(learning_rate=1.,name=name)
        self.param = model.param
        self._trainable_variables = model.param
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        self.local_rank: int = int(os.environ.get('LOCAL_RANK', '0'))

        self.gather_norm: bool = False
        self.clip_coef = None

        p0 = next(iter(self.param))

        self.grad_func = (
            self.fuse_update_zero3() if zero3_enabled else self.fuse_update()
        )

        self.loss_scaler = None
        if p0.dtype == tf.float16:
            if clip_grad_norm is None:
                raise ValueError(
                    '[-] Loss scaling is recommended to be used with grad norm to get better performance.'
                )

            self.loss_scaler = DynamicLossScaler(init_scale=2 ** 16)
        
        self.build(self.param)
    
    def build(self, var_list):
        self.grad_norms = []
        for var in var_list:
            self.grad_norms.append(tf.Variable(var))
            self._track_variable(self.grad_norms[-1])

    def fuse_update(self):
        def func(grads):
            for i, p in enumerate(self.param):
                if not p.trainable or grads[i] is None:
                    continue
                
                def true_fn():
                    grads[i] = None
                    self.loss_scaler.has_overflow_serial = True
                    
                def false_fn():
                    grad_fp32 = tf.cast(grads[i], tf.float32)
                    grads[i] = None
    
                    if self.loss_scaler:
                        grad_fp32 /= self.loss_scaler.loss_scale
    
                    if self.gather_norm:
                        self.grad_norms[i].assign(tf.norm(grad_fp32, 2.0))
                    else:
                        if self.clip_grad_value is not None and self.clip_grad_value > 0.0:
                            grad_fp32 = tf.clip_by_value(grad_fp32, -self.clip_grad_value, self.clip_grad_value)
                        if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0 and self.clip_coef is not None:
                            grad_fp32 *= self.clip_coef
    
                        p_fp32 = tf.cast(p, tf.float32)
                        p_fp32 += grad_fp32 * -self.lr
                        p.assign(p_fp32)
                        
                has_serial_overflow = self.loss_scaler and self.loss_scaler.has_overflow_serial
                nan_mask = tf.math.is_nan(grads[i])
                inf_mask = tf.math.is_inf(grads[i])
                any_bad = tf.reduce_any(tf.math.logical_or(nan_mask, inf_mask))
                tf.cond(tf.logical_or(has_serial_overflow, any_bad), true_fn, false_fn)

            return

        return func
    
    def fuse_update_zero3(self):  # pragma: no cover
        def func(grads):
            for i, p in enumerate(self.param):
                if grads[i] is None:
                    continue
                
                replica_ctx = tf.distribute.get_replica_context()
                grads[i] = replica_ctx.all_reduce(tf.distribute.ReduceOp.MEAN, grads[i])

                def true_fn():
                    grads[i] = None
                    self.loss_scaler.has_overflow_serial = True
                    
                def false_fn():
                    grad_fp32 = tf.cast(grads[i], tf.float32)
                    grads[i] = None
    
                    param_fp32 = tf.cast(p, tf.float32)
                    if self.loss_scaler:
                        grad_fp32 /= self.loss_scaler.loss_scale
    
                    if self.gather_norm:
                        self.grad_norms[i].assign(tf.norm(grad_fp32, 2.0))
                    else:
                        one_dim_grad_fp32 = tf.reshape(grad_fp32, [-1])
    
                        partition_size = tf.size(p)
                        start = partition_size * self.local_rank
                        end = tf.minimum(start + partition_size, tf.size(grad_fp32))
    
                        partitioned_grad_fp32 = nn.narrow(one_dim_grad_fp32, 0, start, end - start)
    
                        if self.clip_grad_value is not None:
                            partitioned_grad_fp32 = tf.clip_by_value(partitioned_grad_fp32, -self.clip_grad_value, self.clip_grad_value)
    
                        if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                            partitioned_grad_fp32 *= self.clip_coef
    
                        partitioned_p = nn.narrow(param_fp32, 0, 0, end - start)
                        partitioned_p += partitioned_grad_fp32 * -self.lr
    
                        p[: end - start].assign(partitioned_p)  # fmt: skip
                    
                has_serial_overflow = self.loss_scaler and self.loss_scaler.has_overflow_serial
                nan_mask = tf.math.is_nan(grads[i])
                inf_mask = tf.math.is_inf(grads[i])
                any_bad = tf.reduce_any(tf.math.logical_or(nan_mask, inf_mask))
                tf.cond(tf.logical_or(has_serial_overflow, any_bad), true_fn, false_fn)

            return

        return func

    def fused_backward(self, tape, loss, variables, lr: float):
        self.lr = lr

        if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0 and self.clip_coef is None:
            raise ValueError(
                'clip_grad_norm is not None, but clip_coef is None. '
                'Please call optimizer.grad_norm() before optimizer.fused_backward().'
            )

        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale

        grads = tape.gradient(loss, variables)

        self.grad_func(grads)
    
    def grad_norm(self, tape, loss, variables):
        self.gather_norm = True

        if self.loss_scaler:
            self.loss_scaler.has_overflow_serial = False
            loss = loss * self.loss_scaler.loss_scale

        grads = tape.gradient(loss, variables)

        self.grad_func(grads)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)
            return

        self.grad_norms = tf.stack(self.grad_norms)

        total_norm = tf.norm(self.grad_norms, 2.0)
        self.clip_coef = tf.clip_by_value(float(self.clip_grad_norm) / (total_norm + 1e-6), clip_value_min=0.0, clip_value_max=1.0)

        self.gather_norm = False
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "clip_grad_norm": self.clip_grad_norm,
                "clip_grad_value": self.clip_grad_value,
                "local_rank": self.local_rank,
                "gather_norm": self.gather_norm,
                "clip_coef": self.clip_coef,
                "grad_func": self.grad_func,
                "loss_scaler": self.loss_scaler,
            }
        )
        return config


def approximate_sq_grad(
    exp_avg_sq_row,
    exp_avg_sq_col,
):
    r"""Get approximation of EMA of squared gradient."""
    r_factor = tf.expand_dims(tf.math.rsqrt((exp_avg_sq_row / tf.reduce_mean(exp_avg_sq_row, axis=-1, keepdims=True))), axis=-1)
    c_factor = tf.math.rsqrt(tf.expand_dims(exp_avg_sq_col, axis=-2))
    return r_factor * c_factor


class AdaLOMO(optimizer.Optimizer):
    def __init__(self, 
                 model,
                 lr=1e-3, 
                 weight_decay: float = 0.0,
                 loss_scale: float = 2.0 ** 10,
                 clip_threshold: float = 1.0,
                 decay_rate: float = -0.8,
                 clip_grad_norm=None, 
                 clip_grad_value=None, 
                 eps1: float = 1e-30,
                 eps2: float = 1e-3,
                 zero3_enabled=True,
                 name="adalomo"):
        super().__init__(learning_rate=1.,name=name)
        self.param = model.param
        self._trainable_variables = model.param
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_scale = loss_scale
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.eps1 = eps1
        self.eps2 = eps2

        self.local_rank: int = int(os.environ.get('LOCAL_RANK', '0'))
        self.zero3_enabled: bool = zero3_enabled

        self.num_steps: int = 0
        self.gather_norm: bool = False
        self.clip_coef = None

        self.grad_func = (
            self.fuse_update_zero3() if zero3_enabled else self.fuse_update()
        )
        
        self.build(self.param)
    
    def initialize_states(self) -> None:
        for i, p in enumerate(self.param):
            if not p.trainable:
                continue
            with tf.device(p.device):
                if self.zero3_enabled:  # pragma: no cover
                    if len(p.shape) == 1:
                        self.exp_avg_sq[i] = tf.Variable(tf.zeros(p.shape[0], dtype=tf.float32))
                        self._track_variable(self.exp_avg_sq[i])
                    else:
                        self.exp_avg_sq_row[i] = tf.Variable(tf.zeros(p.shape[0], dtype=tf.float32))
                        self.exp_avg_sq_col[i] = tf.Variable(tf.zeros(p.shape[1], dtype=tf.float32))
                        self._track_variable(self.exp_avg_sq_row[i])
                        self._track_variable(self.exp_avg_sq_col[i])
                elif len(p.shape) == 1:
                    self.exp_avg_sq[i] = tf.Variable(tf.zeros(p.shape[0], dtype=tf.float32))
                    self._track_variable(self.exp_avg_sq[i])
                else:
                    self.exp_avg_sq_row[i] = tf.Variable(tf.zeros(p.shape[0], dtype=tf.float32))
                    self.exp_avg_sq_col[i] = tf.Variable(tf.zeros(p.shape[1], dtype=tf.float32))
                    self._track_variable(self.exp_avg_sq_row[i])
                    self._track_variable(self.exp_avg_sq_col[i])
    
    def build(self, var_list):
        self.exp_avg_sq = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}
        
        self.initialize_states()
        self.grad_norms = []
        for var in var_list:
            self.grad_norms.append(tf.Variable(var))
            self._track_variable(self.grad_norms[-1])

    def fuse_update(self):
        def func(grads):
            for i, p in enumerate(self.param):
                if not p.trainable or grads[i] is None:
                    continue
                
                grad_fp32 = tf.cast(grads[i], tf.float32)
                grads[i] = None

                if self.loss_scaler:
                    grad_fp32 /= self.loss_scale

                if self.gather_norm:
                    self.grad_norms[i].assign(tf.norm(grad_fp32, 2.0))
                else:
                    if self.clip_grad_value is not None and self.clip_grad_value > 0.0:
                        grad_fp32 = tf.clip_by_value(grad_fp32, -self.clip_grad_value, self.clip_grad_value)
                    if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0 and self.clip_coef is not None:
                        grad_fp32 *= self.clip_coef
                    
                    def true_fn():
                        return 1.0 - tf.pow(self.num_steps, self.decay_rate)
                    def false_fn():
                        return 1.0 - tf.pow(self.num_steps, -self.decay_rate)
                    beta2_t = tf.cond(self.num_steps > 0, true_fn, false_fn)
                    
                    update = tf.pow(grad_fp32, 2) + self.eps1
                    
                    if len(p.shape) > 1:
                        self.exp_avg_sq_row[i].assign(self.exp_avg_sq_row[i] * beta2_t + tf.reduce_mean(update, axis=-1) * (1.0 - beta2_t))
                        self.exp_avg_sq_col[i].assign(self.exp_avg_sq_col[i] * beta2_t + tf.reduce_mean(update, axis=-2) * (1.0 - beta2_t))

                        update = approximate_sq_grad(self.exp_avg_sq_row[i], self.exp_avg_sq_col[i])
                        update *= grad_fp32
                    else:
                        self.exp_avg_sq[i].assign(self.exp_avg_sq[i] * beta2_t + update * (1.0 - beta2_t))
                        update = tf.math.rsqrt(self.exp_avg_sq[i]) * grad_fp32
                    
                    update /= tf.maximum(tf.norm(update, 2) / tf.sqrt(tf.size(update, out_type=update.dtype)) / self.clip_threshold, 1.0)
                    
                    p_fp32 = tf.cast(p, tf.float32)
                    p_rms = tf.norm(p_fp32, 2.0) / tf.sqrt(tf.size(p, out_type=tf.float32))
                    
                    lr = self.lr * tf.maximum(self.eps2, p_rms)
                    
                    if self.weight_decouple:
                        p.assign(p * (1.0 - self.weight_decay * lr))
                    elif self.weight_decay > 0.0:
                        grad_fp32 += p * self.weight_decay
                        
                    p_fp32 += grad_fp32 * -self.lr
                    p.assign(p_fp32)

            return

        return func
    
    def fuse_update_zero3(self):  # pragma: no cover
        def func(grads):
            for i, p in enumerate(self.param):
                if grads[i] is None:
                    continue
                
                replica_ctx = tf.distribute.get_replica_context()
                grads[i] = replica_ctx.all_reduce(tf.distribute.ReduceOp.MEAN, grads[i])
                    
                grad_fp32 = tf.cast(grads[i], tf.float32)
                grads[i] = None
    
                if self.loss_scaler:
                    grad_fp32 /= self.loss_scaler
    
                if self.gather_norm:
                    self.grad_norms[i].assign(tf.norm(grad_fp32, 2.0))
                else:
                    partition_size = tf.size(p)
                    start = partition_size * self.local_rank
                    end = tf.minimum(start + partition_size, tf.size(grad_fp32))
                    
                    if self.clip_grad_value is not None and self.clip_grad_value > 0.0:
                        grad_fp32 = tf.clip_by_value(grad_fp32, -self.clip_grad_value, self.clip_grad_value)
                    if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0 and self.clip_coef is not None:
                        grad_fp32 *= self.clip_coef

                    def true_fn():
                        return 1.0 - tf.pow(self.num_steps, self.decay_rate)
                    def false_fn():
                        return 1.0 - tf.pow(self.num_steps, -self.decay_rate)
                    beta2_t = tf.cond(self.num_steps > 0, true_fn, false_fn)
                    
                    update = tf.pow(grad_fp32, 2) + self.eps1
                    
                    if len(p.shape) > 1:
                        self.exp_avg_sq_row[i].assign(self.exp_avg_sq_row[i] * beta2_t + tf.reduce_mean(update, axis=-1) * (1.0 - beta2_t))
                        self.exp_avg_sq_col[i].assign(self.exp_avg_sq_col[i] * beta2_t + tf.reduce_mean(update, axis=-2) * (1.0 - beta2_t))

                        update = approximate_sq_grad(self.exp_avg_sq_row[i], self.exp_avg_sq_col[i])
                        update *= grad_fp32
                    else:
                        self.exp_avg_sq[i].assign(self.exp_avg_sq[i] * beta2_t + update * (1.0 - beta2_t))
                        update = tf.math.rsqrt(self.exp_avg_sq[i]) * grad_fp32
                    
                    update /= tf.maximum(tf.norm(update, 2) / tf.sqrt(tf.size(update, out_type=update.dtype)) / self.clip_threshold, 1.0)

                    one_dim_update = tf.reshape(update, [-1])
                    partitioned_update = nn.narrow(one_dim_update, 0, start, end - start)
                    
                    param_fp32 = tf.cast(p, tf.float32)
                    partitioned_p = nn.narrow(param_fp32, 0, 0, end - start)

                    p_rms = tf.pow(tf.norm(partitioned_p, 2.0), 2)
                    p_rms = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, p_rms)
                    p_rms = tf.sqrt(p_rms / tf.size(p, out_type=p_rms.dtype))
                    
                    lr = self.lr * tf.maximum(self.eps2, p_rms)
                    
                    if self.weight_decouple:
                        partitioned_p = partitioned_p * (1.0 - self.weight_decay * lr)
                    elif self.weight_decay > 0.0:
                        grad_fp32 += partitioned_p * self.weight_decay

                    partitioned_p += partitioned_update * -lr

                    p[: end - start].assign(partitioned_p)

            return

        return func

    def fused_backward(self, tape, loss, variables, lr: float):
        self.lr = lr

        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale
        
        self.iterations.assign_add(1)
        self.num_steps = tf.cast(self.iterations, tf.float32)

        grads = tape.gradient(loss, variables)

        self.grad_func(grads)
    
    def grad_norm(self, tape, loss, variables):
        self.gather_norm = True

        if self.loss_scaler:
            loss = loss * self.loss_scaler

        grads = tape.gradient(loss, variables)

        self.grad_func(grads)

        self.grad_norms = tf.stack(self.grad_norms)

        total_norm = tf.norm(self.grad_norms, 2.0)
        self.clip_coef = tf.clip_by_value(float(self.clip_grad_norm) / (total_norm + 1e-6), clip_value_min=0.0, clip_value_max=1.0)

        self.gather_norm = False
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "loss_scale": self.loss_scale,
                "clip_threshold": self.clip_threshold,
                "decay_rate": self.decay_rate,
                "clip_grad_norm": self.clip_grad_norm,
                "clip_grad_value": self.clip_grad_value,
                "eps1": self.eps1,
                "eps2": self.eps2,
                "local_rank": self.local_rank,
                "zero3_enabled": self.zero3_enabled,
                "gather_norm": self.gather_norm,
                "clip_coef": self.clip_coef,
                "grad_func": self.grad_func,
            }
        )
        return config


class DynamicLossScaler:
    r"""Dynamically adjusts the loss scaling factor.

        Dynamic loss scalers are important in mixed-precision training.
        They help us avoid underflows and overflows in low-precision gradients.

        See here for information:
        <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling>

        Shamelessly stolen and adapted from FairSeq.
        <https://github.com/pytorch/fairseq/blob/main/fairseq/optim/fp16_optimizer.py>

        Reference : 'https://github.com/facebookresearch/ParlAI/blob/main/parlai/utils/fp16.py'

    :param init_scale: Initial loss scale.
    :param scale_factor: Factor by which to increase or decrease loss scale.
    :param scale_window: If we do not experience overflow in scale_window iterations,
        loss scale will increase by scale_factor.
    :param tolerance: Pct of iterations that have overflowed after which we must decrease the loss scale.
    :param threshold: If not None, loss scale will decrease below this threshold.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 15,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        tolerance: float = 0.00,
        threshold = None,
    ):  # fmt: skip
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold

        self.iter: int = 0
        self.last_overflow_iter: int = -1
        self.last_rescale_iter: int = -1
        self.overflows_since_rescale: int = 0
        self.has_overflow_serial: bool = False

    def update_scale(self, overflow: bool):
        r"""Update the loss scale.

            If overflow exceeds our tolerance, we decrease the loss scale.
            If the number of iterations since the last overflow exceeds the scale window, we increase the loss scale.

        :param overflow: bool. adjust scales to prevent overflow.
        """
        iter_since_rescale: int = self.iter - self.last_rescale_iter

        if overflow:
            # calculate how often we overflowed already
            self.last_overflow_iter = self.iter
            self.overflows_since_rescale += 1

            pct_overflow: float = self.overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                # decrease loss scale by the scale factor
                self.decrease_loss_scale()

                # reset iterations
                self.last_rescale_iter = self.iter
                self.overflows_since_rescale = 0
        elif (self.iter - self.last_overflow_iter) % self.scale_window == 0:
            # increase the loss scale by scale factor
            self.loss_scale *= self.scale_factor
            self.last_rescale_iter = self.iter

        self.iter += 1

    def decrease_loss_scale(self):
        r"""Decrease the loss scale by self.scale_factor.

        NOTE: the loss_scale will not go below `self.threshold`.
        """
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)
