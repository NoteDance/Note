""" NAdamW Optimizer

Based on simplified algorithm in https://github.com/mlcommons/algorithmic-efficiency/tree/main/baselines/nadamw

Added multi-tensor (foreach) path.
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


# Modified from github.com/pytorch/pytorch/blob/v1.12.1/torch/optim/adamw.py.
class NAdamW(optimizer.Optimizer):
    """ Implements NAdamW algorithm.

    See Table 1 in https://arxiv.org/abs/1910.05446 for the implementation of
    the NAdam algorithm (there is also a comment in the code which highlights
    the only difference of NAdamW and AdamW).

    For further details regarding the algorithm we refer to
        - Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        - On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-2,
        caution=False,
        maximize=False,
        foreach=None,
        capturable=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="nadamw",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.caution = caution
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not tf.is_tensor(self.step[0]):
            for p in self._trainable_variables:
                self.step[self._get_variable_index(p)] = tf.convert_to_tensor(float(self.step[self._get_variable_index(p)]))
        self.caution = False

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.step = 0
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        
        for p in trainable_variables:
            exp_avgs.append(self.exp_avg[self._get_variable_index(p)])
            exp_avg_sqs.append(self.exp_avg_sq[self._get_variable_index(p)])
            state_steps.append(self.step)

        nadamw(
            trainable_variables,
            grads,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.epsilon,
            caution=self.caution,
            maximize=self.maximize,
            capturable=self.capturable,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "caution": self.caution,
                "maximize": self.maximize,
                "foreach": self.foreach,
                "capturable": self.capturable,
                "step": self.iterations.numpy(),
            }
        )
        return config
    
    def _update_step(self):
        if hasattr(self, 'step'):
            if type(self.step) == list:
                self.step = [self.iterations.numpy() for _ in range(len(self.step))]
            else:
                self.step = self.iterations.numpy()
	
    def _apply_weight_decay(self, variables):
        pass

def nadamw(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        foreach = None,
        capturable = False,
        *,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
        caution,
        maximize,
):
    r"""Functional API that performs NAdamW algorithm computation.
      See NAdamW class for details.
    """

    if not all(isinstance(t, tf.Tensor) for t in state_steps):
        raise RuntimeError(
            'API has changed, `state_steps` argument must contain a list of' +
            ' singleton tensors')

    if foreach is None:
        foreach = False

    if foreach:
        func = _multi_tensor_nadamw
    else:
        func = _single_tensor_nadamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        caution=caution,
        maximize=maximize,
        capturable=capturable,
    )

def _single_tensor_nadamw(
        params, grads, exp_avgs, exp_avg_sqs, state_steps, *,
        beta1, beta2, lr, weight_decay,
        eps, caution, maximize, capturable):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Update self.step.
        step_t += 1
        state_steps[i] = step_t

        # Perform stepweight decay.
        param.assign(param * (1. - lr * weight_decay))

        # Decay the first and second moment running average coefficient.
        exp_avg.assign(beta1 * exp_avg + (1 - beta1) * grad)
        exp_avg_sq.assign(beta2 * exp_avg_sq + (1 - beta2) * tf.square(grad))

        if capturable:
            step = tf.cast(step_t, param.dtype)
            
            # 1 - beta1 ** self.step can't be captured in a CUDA graph, even if self.step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - tf.pow(beta1, step)
            bias_correction2 = 1 - tf.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = -step_size
            
            bias_correction2_sqrt = tf.sqrt(bias_correction2)
            
            # Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            exp_avg.assign(beta1 * exp_avg + (1 - beta1) * grad)
            
            denom = (tf.sqrt(exp_avg_sq) / (bias_correction2_sqrt * step_size_neg)) + (eps / step_size_neg)

            if caution:
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                # FIXME not 100% sure if this remains capturable?
                mask = tf.cast(exp_avg * grad > 0, grad.dtype)
                mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)
                exp_avg.assign(exp_avg * mask)

            param.assign_add(exp_avg / denom)
        else:
            step = step_t
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            
            # Apply Nesterov. Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            exp_avg.assign(beta1 * exp_avg + (1 - beta1) * grad)
            denom = (tf.sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps

            if caution:
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                mask = tf.cast(exp_avg * grad > 0, grad.dtype)
                mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)
                exp_avg.assign(exp_avg * mask)

            param.assign_add(-step_size * exp_avg / denom)

def _multi_tensor_nadamw(
        params, grads, exp_avgs, exp_avg_sqs, state_steps,
        beta1, beta2, lr, weight_decay, eps, caution, maximize, capturable
):
    if maximize:
        grads = [-grad for grad in grads]  # type: ignore[assignment]
    
    grads = [tf.math.real(x) if x.dtype.is_complex else x for x in grads]
    exp_avgs = [tf.math.real(x) if x.dtype.is_complex else x for x in exp_avgs]
    exp_avg_sqs = [tf.math.real(x) if x.dtype.is_complex else x for x in exp_avg_sqs]
    params = [tf.math.real(x) if x.dtype.is_complex else x for x in params]

    # update steps
    for i in range(len(state_steps)):
        state_steps[i] += 1

    # Perform stepweight decay
    for i in range(len(params)):
        params[i].assign(params[i] * (1 - lr * weight_decay))

    # Decay the first and second moment running average coefficient
    for i in range(len(exp_avgs)):
        exp_avgs[i].assign(exp_avgs[i] * beta1 + grads[i] * (1 - beta1))
        exp_avg_sqs[i].assign(exp_avg_sqs[i] * beta2 + tf.square(grads[i]) * (1 - beta2))

    if capturable:
        bias_correction1 = [tf.pow(beta1, tf.cast(step, p.dtype)) for step, p in zip(state_steps, params)]
        bias_correction2 = [tf.pow(beta2, tf.cast(step, p.dtype)) for step, p in zip(state_steps, params)]

        bias_correction1 = [1 - bc for bc in bias_correction1]
        bias_correction2 = [1 - bc for bc in bias_correction2]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]
        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        # Only difference between NAdamW and AdamW in this implementation.
        # The official PyTorch implementation of NAdam uses a different algorithm.
        for i in range(len(exp_avgs)):
            exp_avgs[i].assign(exp_avgs[i] * beta1 + grads[i] * (1 - beta1))

        denom = [
            tf.sqrt(exp_avg_sqs[i]) / (bias_correction2_sqrt[i] * step_size[i]) + eps / step_size[i]
            for i in range(len(exp_avg_sqs))
        ]

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            for i in range(len(exp_avgs)):
                mask = tf.cast(exp_avgs[i] * grads[i] > 0, grads[i].dtype)
                mask_scale = tf.maximum(tf.reduce_mean(mask), 1e-3)
                mask = mask / mask_scale
                exp_avgs[i].assign(exp_avgs[i] * mask)

        for i in range(len(params)):
            params[i].assign_add(exp_avgs[i] / denom[i])
    else:
        bias_correction1 = [1 - beta1 ** step for step in state_steps]
        bias_correction2 = [1 - beta2 ** step for step in state_steps]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]
        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        # Apply Nesterov. Only difference between NAdamW and AdamW in this implementation.
        # The official PyTorch implementation of NAdam uses a different algorithm.
        for i in range(len(exp_avgs)):
            exp_avgs[i].assign(exp_avgs[i] * beta1 + grads[i] * (1 - beta1))

        denom = [
            tf.sqrt(exp_avg_sqs[i]) / bias_correction2_sqrt[i] + eps
            for i in range(len(exp_avg_sqs))
        ]

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            for i in range(len(exp_avgs)):
                mask = tf.cast(exp_avgs[i] * grads[i] > 0, grads[i].dtype)
                mask_scale = tf.maximum(tf.reduce_mean(mask), 1e-3)
                mask = mask / mask_scale
                exp_avgs[i].assign(exp_avgs[i] * mask)

        for i in range(len(params)):
            params[i].assign_add(exp_avgs[i] / denom[i] * step_size[i])