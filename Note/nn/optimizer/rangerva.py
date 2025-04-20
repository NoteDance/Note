""" RangerVA
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class Softplus:
    def __init__(self, beta=1.0, threshold=20.0):
        """
        TensorFlow implementation of PyTorch's torch.nn.Softplus.
        Args:
            beta (float): Controls the smoothness of the Softplus function.
            threshold (float): Threshold value to avoid overflow for large inputs.
        """
        self.beta = beta
        self.threshold = threshold

    def __call__(self, inputs):
        """
        Forward pass of the Softplus function.
        Args:
            inputs (tf.Tensor): Input tensor.
        Returns:
            tf.Tensor: Softplus-activated tensor.
        """
        if self.beta != 1.0:
            inputs = inputs * self.beta
        result = tf.where(
            inputs > self.threshold,
            inputs,  # Approximation for large inputs to avoid overflow
            tf.math.log(1 + tf.exp(inputs))
        )
        if self.beta != 1.0:
            result = result / self.beta
        return result


class RangerVA(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=.95,
        beta2=0.999,
        epsilon=1e-5,
        weight_decay=0,
        alpha=0.5,
        k=6,
        n_sma_threshhold=5,
        amsgrad=True,
        transformer='softplus',
        smooth=50,
        grad_transformer='square',
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="rangerva",
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
        self.alpha = alpha
        self.k = k
        self.n_sma_threshhold = n_sma_threshhold
        self.amsgrad = amsgrad
        self.transformer = transformer
        self.smooth = smooth
        self.grad_transformer = grad_transformer
        self.radam_buffer = [[None, None, None] for ind in range(10)]

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        if self.amsgrad:
            self.max_exp_avg_sq = []
        self.slow_buffer = []
        self.step = []
        for var in var_list:
            var_fp32 = tf.Variable(tf.cast(var, 'float32'))
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var_fp32, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var_fp32, name="exp_avg_sq"
                )
            )
            if self.amsgrad:
                self.max_exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="max_exp_avg_sq"
                    )
                )  
            self.slow_buffer.append(tf.Variable(var))
            self._track_variable(self.slow_buffer[-1])
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        if gradient.dtype != tf.float32:
            gradient = tf.cast(gradient, 'float32')
        if variable.dtype != tf.float32:
            variable_fp32 = tf.cast(variable, 'float32')
        else:
            variable_fp32 = tf.convert_to_tensor(variable)
        lr = tf.cast(learning_rate, variable_fp32.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'Ranger optimizer does not support sparse gradients')
        
        # begin computations
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        if self.amsgrad:
            max_exp_avg_sq = self.max_exp_avg_sq[self._get_variable_index(variable)]

        # compute variance mov avg
        exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * tf.square(gradient))
        # compute mean moving avg
        exp_avg.assign(self.beta1 * exp_avg + (1 - self.beta1) * gradient)
        
        ##transformer
        if self.grad_transformer == 'square':
            grad_tmp = gradient**2
        elif self.grad_transformer == 'abs':
            grad_tmp = tf.abs(gradient)
        
        exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * grad_tmp)
        
        if self.amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq.assign(tf.maximum(max_exp_avg_sq, exp_avg_sq))
            # Use the max. for normalizing running avg. of gradient
            denomc = tf.identity(max_exp_avg_sq)
        else:
            denomc = tf.identity(exp_avg_sq)
        
        if self.grad_transformer == 'square':
            #pdb.set_trace()
            denomc = tf.sqrt(denomc)
        
        self.step[self._get_variable_index(variable)] += 1
        
        if self.weight_decay != 0:
            variable_fp32 += -self.weight_decay * lr * variable_fp32
        
        bias_correction1 = 1 - self.beta1 ** self.step[self._get_variable_index(variable)]
        bias_correction2 = 1 - self.beta2 ** self.step[self._get_variable_index(variable)]
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1 

        # ...let's use calibrated alr 
        if  self.transformer =='softplus':
            sp = Softplus(self.smooth)
            denomf = sp(denomc)
            variable_fp32 += -step_size * exp_avg / denomf
        else:
            denom = tf.sqrt(exp_avg_sq) + self.epsilon
            variable_fp32 += -step_size * lr * exp_avg / denom

        variable.assign(tf.cast(variable_fp32, variable.dtype))

        # integrated look ahead...
        # we do it at the param level instead of group level
        if self.step[self._get_variable_index(variable)] % self.k == 0:
            # get access to slow param tensor
            slow_p = self.slow_buffer[self._get_variable_index(variable)]
            # (fast weights - slow weights) * alpha
            slow_p.assign_add(self.alpha * (variable - slow_p))
            # copy interpolated weights to RAdam param tensor
            variable.assign(slow_p)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "k": self.k,
                "n_sma_threshhold": self.n_sma_threshhold,
                "amsgrad": self.amsgrad,
                "transformer": self.transformer,
                "smooth": self.smooth,
                "grad_transformer": self.grad_transformer,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass