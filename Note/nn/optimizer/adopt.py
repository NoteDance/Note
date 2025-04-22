""" ADOPT Note Optimizer

ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate: https://arxiv.org/abs/2411.02853

Modified to be compatible with TensorFlow and Keras from original at: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adopt.py

@inproceedings{taniguchi2024adopt,
 author={Taniguchi, Shohei and Harada, Keno and Minegishi, Gouki and Oshima, Yuta and Jeong, Seong Cheol and Nagahara, Go and Iiyama, Tomoshi and Suzuki, Masahiro and Iwasawa, Yusuke and Matsuo, Yutaka},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate},
 year = {2024}
}
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def _view_as_real(params, *state_and_grads):
    for i, p in enumerate(params):
        if tf.is_tensor(p) and p.dtype.is_complex:
            params[i].assign(tf.math.real(p))
            for s in state_and_grads:
                s[i].assign(tf.math.real(s[i]))


def _get_scalar_dtype(is_fused=None):
    if is_fused:
        return tf.float32
    return tf.float64 if tf.keras.backend.floatx() == 'float64' else tf.float32


def _is_compiling():
    return tf.executing_eagerly() == False


def _get_value(x):
    if tf.executing_eagerly():
        return x.numpy() if tf.is_tensor(x) else x
    else:
        return x


class Adopt(optimizer.Optimizer):
    """
    ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate: https://arxiv.org/abs/2411.02853

    """
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.9999,
        epsilon=1e-6,
        weight_decay=0.0,
        clip_exp=0.333,
        decoupled=False,
        caution=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adopt",
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
        self.clip_exp = clip_exp
        self.decoupled = decoupled
        self.caution = caution
        self.foreach = foreach
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.maximize = False
        self.foreach = None
        self.capturable = False
        self.differentiable = False
        self.clip_exp = None
        self.caution = False
        self._iterations.assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
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
    
    def _init_group(
            self,
            trainable_variables,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
    ):
        has_complex = False
        for p in trainable_variables:
            step = tf.cast(self.iterations + 1, p.dtype)
            
            has_complex |= p.dtype.is_complex

            exp_avgs.append(self.exp_avg[self._get_variable_index(p)])
            exp_avg_sqs.append(self.exp_avg_sq[self._get_variable_index(p)])

            state_steps.append(step)
        return has_complex

    def update_step(self, grads, trainable_variables, learning_rate):
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        
        has_complex = self._init_group(
            trainable_variables,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
        )
        
        adopt(
            trainable_variables,
            grads,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            has_complex=has_complex,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            weight_decay=self.weight_decay,
            clip_exp=self.clip_exp,
            decoupled=self.decoupled,
            eps=self.epsilon,
            caution=self.caution,
            maximize=self.maximize,
            foreach=self.foreach,
            capturable=self.capturable,
            differentiable=self.differentiable,
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr, 
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "clip_exp": self.clip_exp,
                "decoupled": self.decoupled,
                "caution": self.caution,
                "foreach": self.foreach,
                "maximize": self.maximize,
                "capturable": self.capturable,
                "differentiable": self.differentiable,
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


def _single_tensor_adopt(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    state_steps,
    grad_scale,
    found_inf,
    *,
    has_complex,
    beta1,
    beta2,
    lr,
    weight_decay,
    clip_exp,
    decoupled,
    eps,
    caution,
    maximize,
    capturable,
    differentiable,
):
    for i, param in enumerate(params):
        grad = grads[i]
        if maximize:
            grad = -grad
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if param.dtype.is_complex:
            grad = tf.math.real(grad)
            param = tf.math.real(param)
            exp_avg = tf.math.real(exp_avg)
            exp_avg_sq = tf.math.real(exp_avg_sq)

        if weight_decay != 0 and not decoupled:
            grad += weight_decay * param

        step = step_t if capturable and differentiable else _get_value(step_t)
        
        def true_fn():
            exp_avg_sq.assign_add(tf.math.conj(grad) * grad)
            
        def false_fn(exp_avg = exp_avg):
            if weight_decay != 0 and decoupled:
                param.assign_add(-lr * weight_decay * param)
    
            denom = tf.maximum(tf.sqrt(exp_avg_sq), eps)
            normed_grad = grad / denom
    
            if clip_exp is not None:
                clip_val = (step - 1) ** clip_exp
                normed_grad = tf.clip_by_value(normed_grad, -clip_val, clip_val)
    
            exp_avg.assign(beta1 * exp_avg + (1 - beta1) * normed_grad)
    
            if caution:
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                mask = tf.cast(exp_avg * grad > 0, grad.dtype)
                mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)
                exp_avg *= mask
    
            param.assign_add(-lr * exp_avg)
    
            exp_avg_sq.assign(beta2 * exp_avg_sq + (1 - beta2) * tf.math.conj(grad) * grad)
            
        tf.cond(step == 1, true_fn, false_fn)


def _multi_tensor_adopt(
        params, grads, exp_avgs, exp_avg_sqs, state_steps, 
        grad_scale, found_inf, *,
        has_complex, beta1, beta2, lr,
        weight_decay, clip_exp, decoupled, eps,
        caution, maximize, capturable, differentiable):
    # Handle complex parameters
    if has_complex:
        _view_as_real(params, grads, exp_avgs, exp_avg_sqs)
    
    if maximize:
        grads = [-g for g in grads]  # type: ignore[assignment]
            
    # Helper functions
    def update_steps(state_steps):
        return [step + 1 for step in state_steps]
    
    def update_grads(params, grads, weight_decay, decoupled):
        if weight_decay != 0:
            if not decoupled:
                for i in range(len(params)):
                    grads[i] = grads[i] + weight_decay * params[i]

    def update_params(params, grads, weight_decay, decoupled):
        if weight_decay != 0:
            if decoupled:
                [p.assign(p - lr * weight_decay * p) for p in params]

    def normalize_grads(grads, exp_avg_sqs, eps, clip_exp, state_steps):
        exp_avg_sq_sqrt = [sq.assign(tf.sqrt(sq)) for sq in exp_avg_sqs]
        [sqrt.assign(tf.maximum(sqrt, eps)) for sqrt in exp_avg_sq_sqrt]
        normed_grad = [g / sqrt for g, sqrt in zip(grads, exp_avg_sq_sqrt)]

        if clip_exp is not None:
            clip_val = (state_steps[0] - 1) ** clip_exp
            for i, g in enumerate(normed_grad):
                normed_grad[i] = tf.clip_by_value(normed_grad[i], -clip_val, clip_val)

        return normed_grad

    def apply_caution(exp_avgs, grads, caution):
        if caution:
            masks = [tf.cast(avg.assign(avg * grad) > 0, grad.dtype) for avg, grad in zip(exp_avgs, grads)]
            mask_scale = [tf.reduce_mean(mask) for mask in masks]
            for i in range(len(mask_scale)):
                mask_scale[i] = tf.maximum(mask_scale, 1e-3)
            for i in range(len(mask_scale)):
                masks[i] = masks[i] / mask_scale[i]
            return [avg * mask for avg, mask in zip(exp_avgs, masks)]
        return exp_avgs

    # Update logic
    state_steps = update_steps(state_steps)
    update_grads(params, grads, weight_decay, decoupled)

    if state_steps[0] == 1:
        [sq.assign_add(grad * grad) for sq, grad in zip(exp_avg_sqs, grads)]
    
    update_params(params, grads, weight_decay, decoupled)

    normed_grad = normalize_grads(grads, exp_avg_sqs, eps, clip_exp, state_steps)
    [avg.assign((1 - beta1) * grad + beta1 * avg) for avg, grad in zip(exp_avgs, normed_grad)]

    if caution:
        exp_avgs = apply_caution(exp_avgs, grads, caution)
    
    [param.assign_add(-lr * avg) for param, avg in zip(params, exp_avgs)]
    [sq.assign(beta2 * sq + (1 - beta2) * grad * grad) for sq, grad in zip(exp_avg_sqs, grads)]


def adopt(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        foreach,
        capturable,
        differentiable,
        grad_scale,
        found_inf,
        has_complex,
        *,
        beta1,
        beta2,
        lr,
        weight_decay,
        clip_exp,
        decoupled,
        eps,
        caution,
        maximize,
):
    r"""Functional API that performs ADOPT algorithm computation.

    """
    if foreach is None:
        foreach = False

    if foreach:
        func = _multi_tensor_adopt
    else:
        func = _single_tensor_adopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        clip_exp=clip_exp,
        decoupled=decoupled,
        eps=eps,
        caution=caution,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )