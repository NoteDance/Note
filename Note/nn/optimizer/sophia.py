""" SophiaH
https://arxiv.org/abs/2305.14342

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class SophiaH(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=6e-2,
        beta1=0.96,
        beta2=0.99,
        epsilon=1e-12,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        p=1e-2,
        update_period=10,
        num_samples=1,
        hessian_distribution='gaussian',
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="sophia",
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
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.p = p
        self.update_period = update_period
        self.num_samples = num_samples
        self.distribution = hessian_distribution
    
    def reset(self):
        iterations = tf.Variable(
                0,
                name="iteration",
                dtype=tf.int64,
                trainable=False,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )
        self._track_variable(iterations)
        self._iterations = iterations
        self.step = 0
        for var in self._trainable_variables:
            self.momentum[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum"
                                                    )
            self.hessian_moment[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="hessian_moment"
                                                    )
            self.hessian[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="hessian"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum = []
        self.hessian_moment = []
        self.hessian = []
        self.step = 0
        for var in var_list:
            self.momentum.append(self.add_variable_from_reference(
                                reference_variable=var, name="momentum"
                                                    ))
            self.hessian_moment.append(self.add_variable_from_reference(
                                reference_variable=var, name="hessian_moment"
                                                    ))
            self.hessian.append(self.add_variable_from_reference(
                                reference_variable=var, name="hessian"
                                                    ))
    
    def apply_gradients(self, grads_and_vars, tape):
        self.tape = tape
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)
    
    def compute_hutchinson_hessian(
        self,
        grads,
        num_samples: int = 1,
        alpha: float = 1.0,
        distribution: str = 'gaussian',
    ) -> None:
        if distribution not in ('gaussian', 'rademacher'):
            raise NotImplementedError(f'hessian with distribution {distribution} is not implemented.')

        params = [p for p in self._trainable_variables if not tf.keras.backend.is_sparse(p)]
        if len(params) == 0:
            return
        
        grads = [g for g in grads if not tf.keras.backend.is_sparse(g)]

        for i in range(num_samples):
            if distribution == 'rademacher':
                zs = [
                    tf.cast(tf.random.uniform(tf.shape(p), 0, 2, dtype=tf.int32)*2 - 1, p.dtype)
                    for p in params
                ]
            else:
                zs = [tf.random.normal(tf.shape(p), dtype=p.dtype) for p in params]

            h_zs = self.tape.gradient(grads, params, zs)

            for h_z, z, p in zip(h_zs, zs, params):
                self.hessian[self._get_variable_index(p)].assign_add(h_z * z * alpha / num_samples)

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.step % self.update_period == 0:
            self.compute_hutchinson_hessian(
                grads,
                num_samples=self.num_samples,
                distribution=self.distribution,
            )
        
        self.step += 1
        
        for p, g in zip(trainable_variables, grads):
            lr = tf.cast(learning_rate, p.dtype)
            
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    'SophiaH does not support sparse gradients')

            if self.weight_decouple:
                p.assign(p * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                g += p * self.weight_decay

            momentum = self.momentum[self._get_variable_index(p)]
            hessian_moment = self.hessian_moment[self._get_variable_index(p)]
            momentum.assign(momentum * self.beta1 + g * (1.0 - self.beta1))

            if self.step % self.update_period == 0:
                hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[self._get_variable_index(p)] * (1.0 - self.beta2))

            update = tf.clip_by_value(momentum / tf.maximum(hessian_moment, self.epsilon), clip_value_min=-p, clip_value_max=p)
            p.assign_add(update * -lr)       

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "p": self.p,
                "update_period": self.update_period,
                "num_samples": self.num_samples,
                "distribution": self.distribution,
                "step": self.iterations.numpy(),
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass