import tensorflow as tf
from typing import Union

from Note.nn.optimizer import optimizer


class ROSE(optimizer.Optimizer):
    """Range-Of-Slice Equilibration optimizer for Keras/TensorFlow.

    Args:
        learning_rate: Learning rate (default 1e-3).
        weight_decay: Weight decay (L2 penalty).
        wd_schedule: Schedule-Coupled Weight Decay.
            - False: standard decoupled weight decay
            - True: uses lr / max_lr (or initial_lr)
            - float: uses that value as lr_ref
        weight_decouple: Use decoupled weight decay (AdamW style).
        fixed_decay: Fix weight decay (don't multiply by lr).
        centralize: Gradient Centralization (for ndim >= 2).
        stabilize: Coefficient-of-Variation Trust Gating.
        maximize: Maximize the objective instead of minimizing.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        wd_schedule: Union[bool, float] = False,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        centralize: bool = True,
        stabilize: bool = True,
        maximize: bool = False,
        name: str = "rose",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )
        self.wd_schedule = wd_schedule
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.centralize = centralize
        self.stabilize = stabilize
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError("ROSE does not support sparse gradients.")

        if variable.dtype.is_complex:
            raise RuntimeError("ROSE does not support complex parameters.")

        lr = tf.cast(learning_rate, variable.dtype)
        grad = tf.cast(gradient, tf.float32)
        param = tf.cast(variable, tf.float32)

        if self.maximize:
            grad = -grad

        wd_lr = self._get_wd_lr(lr)
        grad = self.apply_weight_decay(
            variable, grad, wd_lr
        )

        ndim = grad.shape.rank

        if ndim == 0:
            update = tf.sign(grad)
            param -= lr * update

        elif ndim == 1:
            g_min = tf.reduce_min(grad)
            g_max = tf.reduce_max(grad)
            de_nom = tf.abs(g_max) - g_min
            de_nom = tf.where(de_nom == 0.0, 1.0, de_nom)

            update = grad / de_nom
            param -= lr * update

        else:
            active_axes = list(range(1, ndim))

            if self.centralize:
                mean = tf.reduce_mean(grad, axis=active_axes, keepdims=True)
                grad = grad - mean

            g_max = tf.reduce_max(grad, axis=active_axes, keepdims=True)
            g_min = tf.reduce_min(grad, axis=active_axes, keepdims=True)
            raw_scale = tf.abs(g_max) - g_min

            if self.stabilize:
                std = tf.math.reduce_std(raw_scale, keepdims=True)
                mean_scale = tf.reduce_mean(raw_scale, keepdims=True)

                trust = mean_scale / tf.where(
                    mean_scale == 0.0,
                    1.0,
                    std + mean_scale
                )
                de_nom = mean_scale * (1 - trust) + raw_scale * trust
            else:
                de_nom = raw_scale

            de_nom = tf.where(de_nom == 0.0, 1.0, de_nom)

            update = grad / de_nom
            param -= lr * update

        variable.assign(tf.cast(param, variable.dtype))

    def _get_wd_lr(self, lr):
        if self.weight_decay <= 0:
            return lr

        if self.wd_schedule:
            if isinstance(self.wd_schedule, float):
                ref_lr = self.wd_schedule
            else:
                ref_lr = tf.cast(self.learning_rate, tf.float32)
            return lr / ref_lr
        return lr

    def get_config(self):
        config = super().get_config()
        config.update({
            "wd_schedule": self.wd_schedule,
            "weight_decouple": self.weight_decouple,
            "fixed_decay": self.fixed_decay,
            "centralize": self.centralize,
            "stabilize": self.stabilize,
            "maximize": self.maximize,
        })
        return config


class ROSE_e(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        wd_schedule: Union[bool, float] = False,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        centralize: bool = True,
        stabilize: bool = True,
        orthograd: bool = False,
        agc: bool = False,
        cautious: bool = False,
        trust_ratio: bool = False,
        trust_clip: bool = True,
        lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        maximize: bool = False,
        name: str = "rose_e",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )

        self.wd_schedule = wd_schedule
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.centralize = centralize
        self.stabilize = stabilize
        self.orthograd = orthograd
        self.agc = agc
        self.cautious = cautious
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.lookahead = lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError("ROSE_e does not support sparse gradients.")

        if variable.dtype.is_complex:
            raise RuntimeError("ROSE_e does not support complex parameters.")

        lr = tf.cast(learning_rate, variable.dtype)
        grad = tf.cast(gradient, tf.float32)
        param = tf.cast(variable, tf.float32)

        if self.maximize:
            grad = -grad

        step = tf.cast(self.iterations + 1, tf.float32)

        if self.orthograd:
            self.apply_orthogonal_gradients([variable], [grad])

        if self.agc:
            grad = self.agc(variable, grad)

        grad = self.apply_weight_decay(variable, grad, lr)

        ndim = grad.shape.rank

        if ndim == 0:
            update = tf.sign(grad)
            param -= lr * update

        elif ndim == 1:
            g_min = tf.reduce_min(grad)
            g_max = tf.reduce_max(grad)
            de_nom = tf.abs(g_max) - g_min
            de_nom = tf.where(de_nom == 0.0, 1.0, de_nom)
            update = grad / de_nom
            param -= lr * update
        else:
            active_axes = list(range(1, ndim))

            if self.centralize:
                mean = tf.reduce_mean(grad, axis=active_axes, keepdims=True)
                grad = grad - mean

            g_max = tf.reduce_max(grad, axis=active_axes, keepdims=True)
            g_min = tf.reduce_min(grad, axis=active_axes, keepdims=True)
            raw_scale = tf.abs(g_max) - g_min

            if self.stabilize:
                std = tf.math.reduce_std(raw_scale, keepdims=True)
                mean_scale = tf.reduce_mean(raw_scale, keepdims=True)
                trust = mean_scale / (std + mean_scale + 1e-8)
                de_nom = mean_scale * (1 - trust) + raw_scale * trust
            else:
                de_nom = raw_scale

            de_nom = tf.where(de_nom == 0.0, 1.0, de_nom)
            update = grad / de_nom
            param -= lr * update

        if self.trust_ratio:
            param = self.apply_trust_ratio(variable, param)

        if self.cautious:
            param = self.apply_cautious(param, grad)

        variable.assign_sub(lr * tf.cast(param, variable.dtype))

        if self.lookahead:
            self.lookahead_merge(variable, step)

    def _get_wd_lr(self, lr):
        if not self.weight_decay or not self.wd_schedule:
            return lr
        if isinstance(self.wd_schedule, float):
            ref = self.wd_schedule
        else:
            ref = tf.cast(self.learning_rate, tf.float32)
        return lr / ref

    def get_config(self):
        config = super().get_config()
        config.update({
            "wd_schedule": self.wd_schedule,
            "weight_decouple": self.weight_decouple,
            "fixed_decay": self.fixed_decay,
            "centralize": self.centralize,
            "stabilize": self.stabilize,
            "orthograd": self.orthograd,
            "agc": self.agc,
            "cautious": self.cautious,
            "trust_ratio": self.trust_ratio,
            "trust_clip": self.trust_clip,
            "lookahead": self.lookahead,
            "lookahead_merge_time": self.lookahead_merge_time,
            "lookahead_blending_alpha": self.lookahead_blending_alpha,
            "maximize": self.maximize,
        })
        return config