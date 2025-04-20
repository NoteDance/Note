""" Ranger
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


class Ranger(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=.95,
        beta2=0.999,
        epsilon=1e-5,
        weight_decay=0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5,
        use_gc=True,
        gc_conv_only=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="ranger",
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
        self.N_sma_threshhold = N_sma_threshhold
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.gc_gradient_threshold = 3 if gc_conv_only else 1
        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_gradient_threshold == 1):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_gradient_threshold == 3):
            print(f"GC applied to conv layers only")

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
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
        
        # GC operation for Conv layers and FC layers
        if len(gradient.shape) > self.gc_gradient_threshold:
            gradient = gradient - tf.reduce_mean(gradient, axis=tuple(range(1, len(gradient.shape))), keepdims=True)
        
        self.step[self._get_variable_index(variable)] += 1
        
        # compute variance mov avg
        exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * gradient * gradient)
        # compute mean moving avg
        exp_avg.assign(self.beta1 * exp_avg + (1 - self.beta1) * gradient)
        
        buffered = self.radam_buffer[int(self.step[self._get_variable_index(variable)] % 10)]
        
        if self.step[self._get_variable_index(variable)] == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = self.step[self._get_variable_index(variable)]
            beta2_t = self.beta2 ** self.step[self._get_variable_index(variable)]
            N_sma_max = 2 / (1 - self.beta2) - 1
            N_sma = N_sma_max - 2 * self.step[self._get_variable_index(variable)] * beta2_t / (1 - beta2_t)
            buffered[1] = N_sma

            if N_sma > self.N_sma_threshhold:
                step_size = math.sqrt(
                    (1 - beta2_t)
                    * (N_sma - 4)
                    / (N_sma_max - 4)
                    * (N_sma - 2)
                    / N_sma
                    * N_sma_max
                    / (N_sma_max - 2)
                ) / (1 - self.beta1 ** self.step[self._get_variable_index(variable)])
            else:
                step_size = 1.0 / (1 - self.beta1 ** self.step[self._get_variable_index(variable)])
            buffered[2] = step_size
            
        if self.weight_decay != 0:
            variable_fp32 -= self.weight_decay * lr * variable_fp32

        # apply lr
        if N_sma > self.N_sma_threshhold:
            denom = denom = tf.sqrt(exp_avg_sq) + self.epsilon
            variable_fp32 -= lr * step_size * exp_avg / denom
        else:
            variable_fp32 -= lr * step_size * exp_avg

        variable.assign(tf.cast(variable_fp32, variable.dtype))

        # integrated look ahead...
        # we do it at the param level instead of group level
        if self.step[self._get_variable_index(variable)] % self.k == 0:
            # get access to slow param tensor
            slow_p = self.slow_buffer[self._get_variable_index(variable)]
            # (fast weights - slow weights) * alpha
            slow_p.assign_add(self.alpha * (variable- slow_p))
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
                "N_sma_threshhold": self.N_sma_threshhold,
                "use_gc": self.use_gc,
                "gc_conv_only": self.gc_conv_only,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass