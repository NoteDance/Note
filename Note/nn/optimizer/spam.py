""" SPAM
https://arxiv.org/abs/2501.06842

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class CosineDecay:
    r"""Applies cosine decay to a parameter (death_rate), using
       TensorFlow's built-in `tf.keras.optimizers.schedules.CosineDecay`.

    :param death_rate: float. initial value to be decayed.
    :param t_max: int. maximum number of iterations for the decay.
    :param eta_min: Optional[float]. minimum value of the parameter after decay. defaults to 0.
    """
    def __init__(
        self,
        death_rate: float,
        t_max: int,
        eta_min: float = 0.0,
    ):
        self.initial_rate = death_rate
        self.t_max = t_max
        self.eta_min = eta_min
        # alpha is the fraction of the initial_rate at the end of decay
        alpha = eta_min / death_rate if death_rate != 0 else 0.0
        self.schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=death_rate,
            decay_steps=t_max,
            alpha=alpha,
        )

    def get_death_rate(self, current_step):
        r"""Get the updated rate (death_rate) at the given step.
        
        """
        def true_fn():
            return self.eta_min
        def false_fn():
            return self.schedule(current_step)
        return tf.cond(current_step >= self.t_max, true_fn, false_fn)


class SPAM(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        epsilon=1e-6,
        weight_decay=0.0,
        density=1.0,
        warmup_epoch=50,
        threshold=5000,
        grad_accu_steps=20,
        update_proj_gap=500,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="spam",
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
        self.betas = betas
        self.epsilon = epsilon
        self.density = density
        self.warmup_epoch = warmup_epoch
        self.threshold = threshold
        self.grad_accu_steps = grad_accu_steps
        self.update_proj_gap = update_proj_gap
        self.maximize = maximize
        
        self.warmup = CosineDecay(0.99, self.warmup_epoch)
    
    @staticmethod
    def initialize_random_rank_boolean_tensor(m: int, n: int, density: float):
        r"""Create an (m x n) boolean tensor with `density` fraction of True entries.

        :param m: int. number of rows.
        :param n: int. number of columns.
        :param density: float. fraction of True entries. 1.0 means all True.
        """
        total_elements: int = m * n
        non_zero_count: int = int(density * total_elements)

        tensor = tf.zeros(total_elements, dtype=tf.bool)

        if non_zero_count > 0:
            indices = tf.random.shuffle(tf.range(total_elements))[:non_zero_count]
            updates = tf.ones(non_zero_count, dtype=tf.bool)
            tensor = tf.tensor_scatter_nd_update(tensor, tf.expand_dims(indices, 1), updates)

        return tf.reshape(tensor, [m, n])

    def update_mask_random(self, p, old_mask):
        r"""Update a random mask.

        Create a new random mask with the same density, compute overlap ratio with old_mask, and update the EMA for
        the overlap region.

        """
        new_mask = tf.random.uniform(p.shape) < self.density

        exp_avg_new = tf.zeros_like(p)
        exp_avg_sq_new = tf.zeros_like(p)

        intersect = tf.logical_and(new_mask, old_mask)

        exp_avg_new = tf.where(intersect, self.exp_avg[self._get_variable_index(p)], exp_avg_new)
        exp_avg_sq_new = tf.where(intersect, self.exp_avg_sq[self._get_variable_index(p)], exp_avg_sq_new)

        self.exp_avg[self._get_variable_index(p)].assign(exp_avg_new)
        self.exp_avg_sq[self._get_variable_index(p)].assign(exp_avg_sq_new)

        return new_mask
    
    def update_masks(self) -> None:
        r"""Update masks in each parameter group that has 'density'.

        The new mask is selected randomly, and the overlap ratio with the old mask is printed.
        """
        for p in self._trainable_variables:
            if len(p.shape) == 2:
                self.mask[self._get_variable_index(p)].assign(self.update_mask_random(p, self.mask[self._get_variable_index(p)]))
    
    def reset(self):
        self._iterations.assign(0)
        self.current_step.assign(self.warmup_epoch + 1)
        self.total_step.assign(0)
        for var in self._trainable_variables:
            if len(var.shape) == 2:
                self.mask.assign(self.initialize_random_rank_boolean_tensor(
                    m=var.shape[0],
                    n=var.shape[1],
                    density=self.density,
                ))
                self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(var[self.mask[-1]]), name="exp_avg"
                                                        )
                self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=tf.Variable(var[self.mask[-1]]), name="exp_avg_sq"
                                                        )
            else:
                self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="exp_avg"
                                                        )
                self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="exp_avg_sq"
                                                        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.mask = []
        self.current_step = tf.Variable(self.warmup_epoch + 1)
        self.total_step = tf.Variable(0)
        self._track_variable(self.current_step)
        self._track_variable(self.total_step)
        for var in var_list:
            if len(var.shape) == 2:
                self.mask.append(
                    tf.Variable(self.initialize_random_rank_boolean_tensor(
                        m=var.shape[0],
                        n=var.shape[1],
                        density=self.density,
                    ))
                )
                self.exp_avg.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.Variable(var[self.mask[-1]]), name="exp_avg"
                    )
                )
                self.exp_avg_sq.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.Variable(var[self.mask[-1]]), name="exp_avg_sq"
                    )
                )
                self._track_variable(self.mask[-1])
            else:
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
                self.mask.append(None)

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'RACS does not support sparse gradient.')
        
        if variable.dtype.is_complex:
            raise RuntimeError(
                'RACS does not support complex parameter.')
        
        scale_factor = tf.cast(1.0 - self.warmup.get_death_rate(self.current_step), variable.dtype)
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        beta1, beta2 = self.betas
        
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
        
        step_size = lr * bias_correction2_sq / bias_correction1
        
        if self.maximize:
            gradient = -gradient
        
        if self.mask[self._get_variable_index(variable)] is not None:
            gradient = gradient[self.mask[self._get_variable_index(variable)]]
        
        def true_fn():
            self.exp_avg[self._get_variable_index(variable)].assign(tf.zeros_like(gradient))
            self.exp_avg_sq[self._get_variable_index(variable)].assign(tf.zeros_like(gradient))
        def false_fn():
            pass
        tf.cond((self.total_step + 1) % self.update_proj_gap == 0, true_fn, false_fn)
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        
        if self.threshold != 0:
            current_step = self.total_step + 1
            def true_fn(gradient = gradient):
                mask = tf.pow(gradient, 2) > (self.threshold * exp_avg_sq)
                indices = tf.where(mask)
                updated = tf.sign(gradient[mask]) * tf.sqrt(exp_avg_sq[mask] * self.threshold)
                gradient = tf.tensor_scatter_nd_update(gradient, indices, updated)
            def false_fn():
                pass
            tf.cond(tf.logical_and(current_step >= self.grad_accu_steps, tf.logical_or(self.update_proj_gap == 0, current_step % self.update_proj_gap >= self.grad_accu_steps)), 
                    true_fn, false_fn)
        
        exp_avg.assign(exp_avg * beta1 + gradient * (1.0 - beta1))
        exp_avg_sq.assign(exp_avg_sq * beta2 + gradient * gradient * (1.0 - beta2))
        
        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        if self.mask[self._get_variable_index(variable)] is not None:
            grad_full = tf.zeros_like(variable)
            indices = tf.where(self.mask[self._get_variable_index(variable)])
            grad_full = tf.tensor_scatter_nd_update(grad_full, indices, exp_avg / de_nom)
            variable.assign_add(grad_full * -step_size * scale_factor)
        else:
            variable.assign_add(-step_size * scale_factor * exp_avg / de_nom)
        
        if self.mask[self._get_variable_index(variable)] is not None:
            variable.assign(tf.where(self.mask[self._get_variable_index(variable)],
                       variable * (1.0 - self.weight_decay * lr),
                       variable))
        else:
            variable.assign(variable * (1.0 - self.weight_decay * lr))
        
        self.total_step.assign_add(1)
        self.current_step.assign_add(1)
        
        def true_fn():
            self.update_masks()
            self.current_step.assign(0)
            self.warmup = CosineDecay(0.99, self.warmup_epoch)
        def false_fn():
            pass
        tf.cond(tf.logical_and(self.total_step != 0, (self.total_step + 1) % self.update_proj_gap == 0), 
                true_fn, false_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "density": self.density,
                "warmup_epoch": self.warmup_epoch,
                "threshold": self.threshold,
                "grad_accu_steps": self.grad_accu_steps,
                "update_proj_gap": self.update_proj_gap,
                "maximize": self.maximize,
                "warmup": self.warmup,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass