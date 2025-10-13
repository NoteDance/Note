""" Ranger_e
Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import math


def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    '''credit - https://github.com/Yonghongwei/Gradient-Centralization '''
    if use_gc:
        if gc_conv_only:
            if len(x.shape) > 3:
                mean = tf.reduce_mean(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
                x = x - mean
        else:
            if len(x.shape) > 1:
                mean = tf.reduce_mean(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
                x = x - mean
    return x


def closest_smaller_divisor_of_n_to_k(n, k):
    r"""Get closest smaller divisor of n to k."""
    def true_fn():
        return k
    
    def false_fn():
        def true_fn():
            raise ValueError
        def false_fn():
            pass
        tf.cond(tf.logical_or(n <= 1, k <= 1), true_fn, false_fn)
        closest_smaller_divisor = -7
        for i in tf.range(k, 0, -1):
            def true_fn():
                def true_fn():
                    return i
                def false_fn():
                    return -7
                return tf.cond(closest_smaller_divisor == -7, true_fn, false_fn)
            def false_fn():
                return -7  # pragma: no cover
            closest_smaller_divisor = tf.cond(n % i == 0, true_fn, false_fn)
        return closest_smaller_divisor
    
    closest_smaller_divisor = tf.cond(n % k == 0, true_fn, false_fn)
    
    def true_fn():
        return -1
    def false_fn():
        return closest_smaller_divisor
    closest_smaller_divisor = tf.cond(closest_smaller_divisor == -7, true_fn, false_fn)
    
    return closest_smaller_divisor


class Ranger_e(optimizer.Optimizer):
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
        gc_loc=True,
        subset_size=-1,
        sn=True,
        d0=1e-6,
        growth_rate=float('inf'),
        DAdapt=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="ranger2020_e",
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
        self.alpha = alpha
        self.k = k
        self.N_sma_threshhold = N_sma_threshhold
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        self.gc_loc = gc_loc
        self.subset_size = subset_size
        self.sn = sn
        self.d0 = d0
        self.growth_rate = growth_rate
        self.DAdapt = DAdapt
        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.slow_buffer = []
        self.subset_size_ = []
        if self.DAdapt:
            self.s = []
            self.sk_l1 = tf.Variable(0.0)
            self.numerator_acc = tf.Variable(0.0)
            self.numerator_weighted = tf.Variable(0.0)
            self.d0_ = tf.Variable(self.d0)
            self._track_variable(self.sk_l1)
            self._track_variable(self.numerator_acc)
            self._track_variable(self.numerator_weighted)
            self._track_variable(self.d0_)
        for var in var_list:
            var_fp32 = tf.Variable(tf.cast(var, 'float32'))
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var_fp32, name="exp_avg"
                )
            )
            if self.sn:
                size = tf.size(var)
                
                def true_fn():
                    return self.subset_size
                def false_fn():
                    return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                    size,
                    tf.cond(self.subset_size > 0, true_fn, false_fn)
                ))

                reshaped_grad = tf.reshape(var, (size // self.subset_size_[-1], self.subset_size_[-1]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
                second_moment_update = tf.Variable(second_moment_update)
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="exp_avg_sq"
                    ))
            else:
                self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=var, name="exp_avg_sq"
                    ))
            self.slow_buffer.append(tf.Variable(var))
            self._track_variable(self.slow_buffer[-1])
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                                    reference_variable=var, name="s"
                                                        ))
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        for variable, gradient in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(gradient):
                raise RuntimeError(
                    'Ranger optimizer does not support sparse gradients')
                
            if gradient.dtype != tf.float32:
                gradient = tf.cast(gradient, 'float32')
            if variable.dtype != tf.float32:
                variable_fp32 = tf.cast(variable, 'float32')
            else:
                variable_fp32 = tf.convert_to_tensor(variable)
            lr = tf.cast(learning_rate, variable_fp32.dtype)
            
            size = tf.size(gradient)
            
            if self.DAdapt:
                # it's not Adam Debias
                d_lr = self.d0_ * lr
            
            # begin computations
            exp_avg = self.exp_avg[self._get_variable_index(variable)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
            if self.DAdapt:
                s = self.s[self._get_variable_index(variable)]
                if self.sn:
                    s = tf.reshape(s, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
            
                denom = tf.sqrt(exp_avg_sq) + self.epsilon
                flat_grad = tf.reshape(gradient, [-1])
                flat_div = tf.reshape(tf.divide(s, denom), [-1])
                dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
                self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
                
                d_lr = tf.cast(d_lr, dtype=variable.dtype)
            
            # GC operation for Conv layers and FC layers
            # if len(gradient.shape) > self.gc_gradient_threshold:
            #     gradient = gradient - tf.reduce_mean(gradient, axis=tuple(range(1, len(gradient.shape))), keepdims=True)
            if self.gc_loc:
                gradient = centralized_gradient(gradient, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
    
            step = tf.cast(self.iterations + 1, variable_fp32.dtype)
            
            if self.sn:
                reshaped_grad = tf.reshape(gradient, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
            else:
                second_moment_update = tf.pow(gradient, 2)
    
            # compute variance mov avg
            exp_avg_sq.assign(self.beta2 * exp_avg_sq + (1 - self.beta2) * second_moment_update)
    
            # compute mean moving avg
            if self.DAdapt:
                beta2_sq = math.sqrt(self.beta2)
                exp_avg.assign(self.beta1 * exp_avg + d_lr * (1 - self.beta1) * gradient)
                s.assign(s * beta2_sq + gradient * d_lr * (1.0 - beta2_sq))
                self.sk_l1.assign_add(tf.cast(tf.reduce_sum(tf.abs(s)), tf.float32))
            else:
                exp_avg.assign(self.beta1 * exp_avg + (1 - self.beta1) * gradient)

            if not self.DAdapt:
                beta2_t = self.beta2 ** step
                N_sma_max = 2 / (1 - self.beta2) - 1
                N_sma = N_sma_max - 2 * step * beta2_t / (1 - beta2_t)
                
                step_size = tf.sqrt(
                        (1 - beta2_t)
                        * (N_sma - 4)
                        / (N_sma_max - 4)
                        * (N_sma - 2)
                        / N_sma
                        * N_sma_max
                        / (N_sma_max - 2)
                    ) / (1 - self.beta1 ** step)
                
                def true_fn():
                    denom = denom = tf.sqrt(exp_avg_sq) + self.epsilon
                    if self.sn:
                        numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
                        normed_grad = tf.reshape((numerator / denom), variable.shape)
                        G_grad = normed_grad
                    else:
                        G_grad = exp_avg / denom
                    return G_grad
                
                def false_fn():
                    return exp_avg
        
                # if self.weight_decay != 0:
                #     variable_fp32.assign_sub(self.weight_decay * lr * variable_fp32)
        
                # apply lr
                G_grad = tf.cond(N_sma > self.N_sma_threshhold, true_fn, false_fn)
        
                if self.weight_decay != 0:
                    G_grad += self.weight_decay * variable_fp32
                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
        
                variable_fp32 += -step_size * lr * G_grad
                variable.assign(tf.cast(variable_fp32, variable.dtype))
        
                # integrated look ahead...
                # we do it at the param level instead of group level
                def true_fn():
                    # get access to slow param tensor
                    slow_p = self.slow_buffer[self._get_variable_index(variable)]
                    # (fast weights - slow weights) * alpha
                    slow_p.assign_add(self.alpha * (variable- slow_p))
                    # copy interpolated weights to RAdam param tensor
                    variable.assign(slow_p)
                
                def false_fn():
                    pass
                
                tf.cond(step % self.k == 0, true_fn, false_fn)
            
            if self.DAdapt:
                def update_fn():
                    d_lr = self.d0_ * learning_rate
                    
                    beta2_sq = math.sqrt(self.beta2)
                    
                    d = self.d0_
                    self.numerator_weighted.assign(self.numerator_weighted * beta2_sq + self.numerator_acc * (1.0 - beta2_sq))  # fmt: skip
                    
                    if self.lr > 0.0:
                        d_hat = self.numerator_weighted / (1.0 - beta2_sq) * self.sk_l1
                        d = tf.maximum(self.d0_, tf.minimum(d_hat, self.d0_ * self.growth_rate))
                    
                    self.d0_.assign(d)
                    
                    for variable in zip(trainable_variables):
                        if variable.dtype != tf.float32:
                            variable_fp32 = tf.cast(variable, 'float32')
                        else:
                            variable_fp32 = tf.convert_to_tensor(variable)
                        
                        lr = tf.cast(d_lr, variable_fp32.dtype)
                            
                        step = tf.cast(self.iterations + 1, variable_fp32.dtype)
                        
                        beta2_t = self.beta2 ** step
                        N_sma_max = 2 / (1 - self.beta2) - 1
                        N_sma = N_sma_max - 2 * step * beta2_t / (1 - beta2_t)
                        
                        step_size = tf.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            ) / (1 - self.beta1 ** step)
                        
                        exp_avg = self.exp_avg[self._get_variable_index(variable)]
                        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
                        
                        def true_fn():
                            denom = denom = tf.sqrt(exp_avg_sq) + self.epsilon
                            if self.sn:
                                numerator = tf.reshape(exp_avg, (size // self.subset_size_[self._get_variable_index(variable)], self.subset_size_[self._get_variable_index(variable)]))
                                normed_grad = tf.reshape((numerator / denom), variable.shape)
                                G_grad = normed_grad
                            else:
                                G_grad = exp_avg / denom
                            return G_grad
                        
                        def false_fn():
                            return exp_avg
                
                        # if self.weight_decay != 0:
                        #     variable_fp32.assign_sub(self.weight_decay * lr * variable_fp32)
                
                        # apply lr
                        G_grad = tf.cond(N_sma > self.N_sma_threshhold, true_fn, false_fn)
                        G_grad = tf.cast(G_grad, variable_fp32.dtype)
                
                        if self.weight_decay != 0:
                            G_grad += self.weight_decay * variable_fp32
                        # GC operation
                        if self.gc_loc == False:
                            G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
                
                        variable_fp32 += -step_size * lr * G_grad
                        variable.assign(tf.cast(variable_fp32, variable.dtype))
                
                        # integrated look ahead...
                        # we do it at the param level instead of group level
                        def true_fn():
                            # get access to slow param tensor
                            slow_p = self.slow_buffer[self._get_variable_index(variable)]
                            # (fast weights - slow weights) * alpha
                            slow_p.assign_add(self.alpha * (variable- slow_p))
                            # copy interpolated weights to RAdam param tensor
                            variable.assign(slow_p)
                        
                        def false_fn():
                            pass
                        
                        tf.cond(step % self.k == 0, true_fn, false_fn)
                
                def no_update_fn():
                    pass
                
                tf.cond(self.sk_l1 == 0, no_update_fn, update_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "k": self.k,
                "N_sma_threshhold": self.N_sma_threshhold,
                "use_gc": self.use_gc,
                "gc_conv_only": self.gc_conv_only,
                "gc_loc": self.gc_loc,
                "subset_size": self.subset_size,
                "sn": self.sn,
                "d0": self.d0,
                "growth_rate": self.growth_rate,
                "DAdapt": self.DAdapt,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass