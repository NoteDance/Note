""" SGDW
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class SGDW(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay= 0.,
        momentum=0.,
        dampening=0.,
        nesterov=False,
        caution=False,
        maximize=False,
        foreach=None,
        differentiable=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sgdw",
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
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.caution = caution
        self.maximize = maximize
        self.foreach = foreach
        self.differentiable = differentiable
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.caution = False
        self.nesterov = False
        self.maximize = False
        self.foreach = None
        self.differentiable = False

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = dict()
        self.momentum_buffer_list = []
        for var in var_list:
            self.momentum_buffer[self._get_variable_index(var)] = None
            self.momentum_buffer_list.append(None)
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = learning_rate
        
        has_sparse_grad = False
        for i,var in enumerate(trainable_variables):
            if tf.keras.backend.is_sparse(grads[i]):
                has_sparse_grad = True
            
            self.momentum_buffer_list[self._get_variable_index(var)] = self.momentum_buffer[self._get_variable_index(var)]

        sgdw(
            trainable_variables,
            grads,
            self.momentum_buffer_list,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            lr=lr,
            dampening=self.dampening,
            nesterov=self.nesterov,
            caution=self.caution,
            maximize=self.maximize,
            has_sparse_grad=has_sparse_grad,
            foreach=self.foreach,
        )

        # update momentum_buffers in state
        for p, momentum_buffer in zip(trainable_variables, self.momentum_buffer_list):
            self.momentum_buffer[self._get_variable_index(p)] = momentum_buffer
            if tf.get_static_value(self.iterations) == 0:
                self._track_variable(momentum_buffer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "dampening": self.dampening,
                "nesterov": self.nesterov,
                "caution": self.caution,
                "maximize": self.maximize,
                "foreach": self.foreach,
                "differentiable": self.differentiable,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
    
    
def sgdw(
        params,
        grads,
        momentum_buffer_list,
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad,
        foreach,
        *,
        weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        caution,
        maximize
):
    if foreach:
        func = _multi_tensor_sgdw
    else:
        func = _single_tensor_sgdw

    func(
        params,
        grads,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        caution=caution,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
    )


def _single_tensor_sgdw(
    params,
    grads,
    momentum_buffer_list,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    caution,
    maximize,
    has_sparse_grad
):
    for i, param in enumerate(params):
        grad = grads[i]

        if maximize:
            grad = -grad

        param.assign(param * (1. - lr * weight_decay))

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = tf.Variable(grad)
                momentum_buffer_list[i] = buf
            else:
                buf.assign(buf * momentum + grad * (1 - dampening))

            if caution:
                if nesterov:
                    buf = grad + buf * momentum
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                mask = tf.cast(buf * grad > 0, grad.dtype)
                mask = mask / tf.maximum(tf.reduce_mean(mask), 1e-3)
                grad = buf * mask
            else:
                if nesterov:
                    grad = grad + buf * momentum
                else:
                    grad = buf

        param.assign_add(-lr * grad)


def _multi_tensor_sgdw(
    params,
    grads,
    momentum_buffer_list,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    caution,
    maximize,
    has_sparse_grad
):
    if maximize:
        grads = [-g for g in grads]

    for i, param in enumerate(params):
        param.assign(param * (1. - lr * weight_decay))

    if momentum != 0:
        bufs = []
        all_states_with_momentum_buffer = all(buf is not None for buf in momentum_buffer_list)

        if all_states_with_momentum_buffer:
            for buf, grad in zip(momentum_buffer_list, grads):
                buf.assign(buf * momentum + grad * (1 - dampening))
                bufs.append(buf)
        else:
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = tf.Variable(grads[i])
                    momentum_buffer_list[i] = buf
                else:
                    buf = momentum_buffer_list[i]
                    buf.assign(buf * momentum + grads[i] * (1 - dampening))
                bufs.append(buf)

        if caution:
            if nesterov:
                # Can't do nesterov in-place if we want to compare against orig grad for caution
                bufs = [g + buf * momentum for g, buf in zip(grads, bufs)]
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            masks = [buf * g for buf, g in zip(bufs, grads)]
            masks = [tf.cast(m > 0, dtype=g.dtype) for m, g in zip(masks, grads)]
            mask_scale = [tf.reduce_mean(mask) for mask in masks]
            mask_scale = [tf.maximum(scale, 1e-3) for scale in mask_scale]
            masks = [mask / scale for mask, scale in zip(masks, mask_scale)]
            grads = [buf * mask for buf, mask in zip(bufs, masks)]
        else:
            if nesterov:
                grads = [g + buf * momentum for g, buf in zip(grads, bufs)]
            else:
                grads = bufs

    if not has_sparse_grad:
        for param, grad in zip(params, grads):
            param.assign_add(-lr * grad)
    else:
        for param, grad in zip(params, grads):
            if tf.keras.backend.is_sparse(grad):
                tf.tensor_scatter_nd_add(param, grad.indices, -lr * grad.values)
            else:
                param.assign_add(-lr * grad)