import tensorflow as tf
from keras.src.optimizers import optimizer
import math


def _mars_single_tensor_step(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        weight_decay,
        beta1,
        beta2,
        last_grad,
        eps,
        step,
        gamma,
        mars_type,
        is_grad_2d,
        optimize_1d,
        lr_1d_factor,
        betas_1d,
        caution,
):
    # optimize_1d ==> use MARS for 1d param, else use AdamW
    if optimize_1d or is_grad_2d:
        one_minus_beta1 = 1. - beta1
        
        if step == 1:
            # this is a timm addition, making first self.step more consistent when no grad history, otherwise tests fail
            c_t = grad
        else:
            c_t = gamma * (beta1 / one_minus_beta1) * (grad - last_grad) + grad
            c_t_norm = tf.norm(c_t)
            if c_t_norm > 1.0:
                c_t = c_t / c_t_norm

        exp_avg.assign(beta1 * exp_avg + one_minus_beta1 * c_t)

        if caution:
            mask = tf.cast(exp_avg * grad > 0, grad.dtype)
            mask /= tf.clip_by_value(tf.reduce_mean(mask), 1e-3, float('inf'))
            exp_avg *= mask

        if mars_type == "adamw":
            exp_avg_sq.assign(beta2 * exp_avg_sq + (1. - beta2) * tf.square(c_t))
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            denom = (tf.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps
            update = p * weight_decay + (exp_avg / bias_correction1) / denom
        elif mars_type == "lion":
            update = p * weight_decay + tf.sign(exp_avg)
        else:
            raise ValueError("Invalid mars_type")

        p.assign_add(-lr * update)
    else:
        beta1_1d, beta2_1d = betas_1d

        exp_avg.assign(beta1_1d * exp_avg + (1. - beta1_1d) * grad)
        exp_avg_sq.assign(beta2_1d * exp_avg_sq + (1. - beta2_1d) * tf.square(grad))
        
        bias_correction1 = 1.0 - beta1_1d ** step
        bias_correction2 = 1.0 - beta2_1d ** step
        denom = (tf.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps

        if caution:
            mask = tf.cast(exp_avg * grad > 0, grad.dtype)
            mask /= tf.clip_by_value(tf.reduce_mean(mask), 1e-3, float('inf'))
            exp_avg *= mask

        update = p * weight_decay + (exp_avg / bias_correction1) / denom
        p.assign_add(-(lr * lr_1d_factor) * update)

    return exp_avg, exp_avg_sq


class Mars(optimizer.Optimizer):
    """ MARS Optimizer

    Paper: MARS: Unleashing the Power of Variance Reduction for Training Large Models
        https://arxiv.org/abs/2411.10438

    """
    def __init__(
        self,
        learning_rate=3e-3,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
        weight_decay=0,
        gamma=0.025,
        mars_type="adamw",
        optimize_1d=False,
        lr_1d_factor=1.0,
        betas_1d=None,
        caution=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="mars",
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gamma=gamma
        self.mars_type=mars_type
        self.optimize_1d=optimize_1d
        self.lr_1d_factor=lr_1d_factor
        self.betas_1d=betas_1d
        self.caution=caution

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.last_grad = []
        self.exp_avg_sq = []
        self.step = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.last_grad.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="last_grad"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )
            self.step.append(0)
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.caution = False

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        self.step[self._get_variable_index(variable)] += 1
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        last_grad = self.last_grad[self._get_variable_index(variable)]
        beta1, beta2 = self.beta1, self.beta2
        is_grad_2d = gradient.shape.ndims >= 2

        # FIXME add multi-tensor (if usage warrants), make more standard
        _mars_single_tensor_step(
            variable,
            gradient,
            exp_avg,
            exp_avg_sq,
            lr,
            self.weight_decay,
            beta1,
            beta2,
            last_grad,
            self.epsilon,
            self.step[self._get_variable_index(variable)],
            self.gamma,
            mars_type=self.mars_type,
            is_grad_2d=is_grad_2d,
            optimize_1d=self.optimize_1d,
            lr_1d_factor=self.lr_1d_factor,
            betas_1d=self.betas_1d,
            caution=self.caution,
        )
        
        self.last_grad[self._get_variable_index(variable)] = gradient

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "mars_type": self.mars_type,
                "optimize_1d": self.optimize_1d,
                "lr_1d_factor": self.lr_1d_factor,
                "betas_1d": self.betas_1d,
                "caution": self.caution,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass