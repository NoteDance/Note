""" Adafactor (Big Vision variant)

Adapted from the implementation in big vision: https://github.com/google-research/big_vision

Described in 'Scaling Vision Transformers': https://arxiv.org/abs/2106.04560

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def _get_scalar_dtype():
    """Get the scalar dtype that the optimizer uses for state"""
    return tf.float64


def _factored_dims(
        shape,
        factored,
        min_dim_size_to_factor
):
    """Whether to use a factored second moment estimator.

    This function returns a tuple with the two largest axes to reduce over.
    If no two dimensions have size >= min_dim_size_to_factor, return None.

    Args:
      shape: an input shape
      factored: whether to use factored second-moment estimator for > 2d vars.
      min_dim_size_to_factor: only factor accumulator if two array dimensions have at least this size.

    Returns:
      None or a tuple of ints
    """
    if not factored or len(shape) < 2:
        return None
    sorted_dims = sorted(((x, i) for i, x in enumerate(shape)))
    if shape[sorted_dims[-2][1]] < min_dim_size_to_factor:
        return None
    return int(sorted_dims[-2][1]), int(sorted_dims[-1][1])


class AdafactorBigVision(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        epsilon=None,
        weight_decay=0.0,
        min_dim_size_to_factor=16,
        decay_rate=0.8,
        decay_offset=0,
        beta2_cap=0.999,
        momentum=0.9,
        momentum_dtype=tf.bfloat16,
        clipping_threshold=None,
        unscaled_wd=False,
        caution=False,
        foreach=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adafactor_bv",
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
        self.epsilon = epsilon
        self.min_dim_size_to_factor = min_dim_size_to_factor
        self.decay_rate = decay_rate
        self.decay_offset = decay_offset
        self.beta2_cap = beta2_cap
        self.momentum = momentum
        self.momentum_dtype = momentum_dtype
        self.clipping_threshold = clipping_threshold
        self.unscaled_wd = unscaled_wd
        self.caution = caution
        self.foreach = foreach
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.caution = False
        self.foreach = None
        for p in self._trainable_variables:
            if not tf.is_tensor(self.step[self._get_variable_index(p)]):
                self.step[self._get_variable_index(p)] = tf.convert_to_tensor(float(self.step[self._get_variable_index(p)]), dtype=_get_scalar_dtype())
            
            if len(self.exp_avg) != 0 and tf.is_tensor(self.exp_avg[self._get_variable_index(p)]):
                self.exp_avg[self._get_variable_index(p)] = tf.cast(self.exp_avg[self._get_variable_index(p)], dtype=self.momentum_dtype)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.exp_avg_sq_r = []
        self.exp_avg_sq_c = []
        for var in var_list:
            # NOTE self.step on CPU, probably need some more though to make capturable
            self.step.append(tf.convert_to_tensor(0.0, dtype=_get_scalar_dtype()))
            
            shape = var.shape
            factored_dims = _factored_dims(
                shape,
                factored=True,
                min_dim_size_to_factor=self.min_dim_size_to_factor
            )
            
            if factored_dims is not None:
                dc, dr = factored_dims
                row_shape = list(var.shape)
                row_shape[dr] = 1
                col_shape = list(var.shape)
                col_shape[dc] = 1
                self.exp_avg_sq_r.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.zeros(row_shape, dtype=var.dtype), name="exp_avg_sq_r"
                    )
                )
                self.exp_avg_sq_c.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.zeros(col_shape, dtype=var.dtype), name="exp_avg_sq_c"
                    )
                )
            else:
                self.exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="exp_avg_sq"
                    )
                )

            if self.momentum is not None:
                self.exp_avg.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.zeros_like(var, dtype=self.momentum_dtype), name="exp_avg"
                    )
                )
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = learning_rate
        
        exp_avg_sq_rs = []
        exp_avg_sq_cs = []
        exp_avg_sqs = []
        state_steps = []
        exp_avgs = []  # For momentum
        
        for p in trainable_variables:
            state_steps.append(self.step[self._get_variable_index(p)])
            if len(self.exp_avg_sq_r) != 0:
                exp_avg_sq_rs.append(self.exp_avg_sq_r[self._get_variable_index(p)])
                exp_avg_sq_cs.append(self.exp_avg_sq_c[self._get_variable_index(p)])
            else:
                exp_avg_sqs.append(self.exp_avg_sq[self._get_variable_index(p)])
            if self.momentum is not None:
                exp_avgs.append(self.exp_avg[self._get_variable_index(p)])

        if self.foreach:
            func = _multi_tensor_adafactor
        else:
            func = _single_tensor_adafactor

        func(
            params=trainable_variables,
            grads=grads,
            exp_avg_sq_rs=exp_avg_sq_rs,
            exp_avg_sq_cs=exp_avg_sq_cs,
            exp_avg_sqs=exp_avg_sqs,
            exp_avgs=exp_avgs,
            state_steps=state_steps,
            beta2_decay=self.decay_rate,
            beta2_cap=self.beta2_cap,
            min_dim_size_to_factor=self.min_dim_size_to_factor,
            eps=self.epsilon,
            lr=lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            momentum_dtype=self.momentum_dtype,
            clipping_threshold=self.clipping_threshold,
            unscaled_wd=self.unscaled_wd,
            caution=self.caution,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "min_dim_size_to_factor": self.min_dim_size_to_factor,
                "decay_rate": self.decay_rate,
                "decay_offset": self.decay_offset,
                "beta2_cap": self.beta2_cap,
                "momentum": self.momentum,
                "momentum_dtype": self.momentum_dtype,
                "clipping_threshold": self.clipping_threshold,
                "unscaled_wd": self.unscaled_wd,
                "caution": self.caution,
                "foreach": self.foreach,
                "step": self.step,
            }
        )
        return config
		
    def _apply_weight_decay(self, variables):
        pass	
    
def _single_tensor_adafactor(
        params,
        grads,
        exp_avg_sq_rs,
        exp_avg_sq_cs,
        exp_avg_sqs,
        exp_avgs,
        state_steps,
        *,
        beta2_decay,
        beta2_cap,
        min_dim_size_to_factor,
        eps,
        lr,
        weight_decay,
        momentum,
        momentum_dtype,
        clipping_threshold,
        unscaled_wd,
        caution,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg_sq_r = exp_avg_sq_rs[i]
        exp_avg_sq_c = exp_avg_sq_cs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg = exp_avgs[i]
        step_t = state_steps[i]
        if eps is None:
            # default eps for avoiding div by zero, diff from float type eps
            eps = 1e-7 if grad.dtype == tf.float16 else 1e-30

        # Update self.step
        step_t += 1
        beta2_t = min(beta2_cap, 1.0 - float(tf.get_static_value(step_t)) ** (-beta2_decay))
        one_minus_beta2_t = 1 - beta2_t

        grad_sqr = tf.square(grad) + eps
        # NOTE application of eps (epsilon1) mirrors the optax/big vision/t5x approach
        if exp_avg_sq is None:
            # factorized second moment
            dc, dr = _factored_dims(grad.shape, True, min_dim_size_to_factor=min_dim_size_to_factor)
            exp_avg_sq_r.assign(
                exp_avg_sq_r * (1.0 - one_minus_beta2_t) + tf.reduce_mean(grad_sqr, axis=dr, keepdims=True) * one_minus_beta2_t
            )
            exp_avg_sq_c.assign(
                exp_avg_sq_c * (1.0 - one_minus_beta2_t) + tf.reduce_mean(grad_sqr, axis=dc, keepdims=True) * one_minus_beta2_t
            )

            reduce_dc = dc - 1 if dc > dr else dc
            row_col_mean = tf.reduce_mean(exp_avg_sq_r, axis=reduce_dc, keepdims=True)
            row_factor = tf.math.rsqrt(exp_avg_sq_r / row_col_mean)
            col_factor = tf.math.rsqrt(exp_avg_sq_c)

            update = grad * row_factor * col_factor
        else:
            # non-factorized second moment
            assert exp_avg_sq_r is None and exp_avg_sq_c is None
            exp_avg_sq.assign(
                exp_avg_sq * (1.0 - one_minus_beta2_t) + grad_sqr * one_minus_beta2_t
            )
            update = grad * tf.math.rsqrt(exp_avg_sq)

        # Clip by RMS value
        if clipping_threshold is not None:
            rms_norm = tf.norm(update) / tf.sqrt(tf.cast(tf.size(update), grad.dtype))
            denom = tf.minimum(rms_norm / clipping_threshold, 1.0)
            update /= denom

        # Apply momentum (in different dtype)
        if momentum is not None and exp_avg is not None:
            if momentum_dtype != grad.dtype:
                exp_avg.assign(
                    exp_avg * momentum + tf.cast(update, momentum_dtype) * (1 - momentum)
                )
                update = tf.cast(exp_avg, grad.dtype)
            else:
                exp_avg.assign(
                    exp_avg * momentum + update * (1 - momentum)
                )
                update = tf.identity(exp_avg)

            if caution:
                # apply caution as per 'Cautious Optimizers': https://arxiv.org/abs/2411.16085
                mask = tf.cast(update * grad > 0, grad.dtype)
                mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)
                update *= mask

        # Scale by learning rate
        update *= lr

        # Perform weight decay
        if weight_decay != 0:
            if unscaled_wd:
                # match big vision impl, 'fully decoupled' decay w/o LR scaling
                param.assign(param * (1. - weight_decay))
            else:
                # match typical pytorch behaviour for decoupled decay, eg adamw where wd is scaled by LR
                param.assign(param * (1. - lr * weight_decay))

        # Update parameters
        param.assign_add(update * -1.0)

def _multi_tensor_adafactor(
        params,
        grads,
        exp_avg_sq_rs,
        exp_avg_sq_cs,
        exp_avg_sqs,
        exp_avgs,
        state_steps,
        *,
        beta2_decay,
        beta2_cap,
        min_dim_size_to_factor,
        eps,
        lr,
        weight_decay,
        momentum,
        momentum_dtype,
        clipping_threshold,
        unscaled_wd,
        caution,
):
    # FIXME TODO
    assert False, 'multi-tensor fn (foreach=True) not implemented yet'