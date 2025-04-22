""" SAM
https://arxiv.org/abs/2010.01412

Copyright 2025 NoteDance
"""
import tensorflow as tf
from Note import nn
from keras.src.optimizers import optimizer
from contextlib import ExitStack


class SAM(optimizer.Optimizer):
    def __init__(
        self,
        base_optimizer,
        rho=0.05,
        adaptive=False,
        use_gc=False,
        perturb_eps=1e-12,
        name="sam",
        **kwargs,
    ):
        super().__init__(learning_rate=1.,name=name)
        self.base_optimizer = base_optimizer
        self.base_optimizer_ = tf.keras.optimizers.serialize(base_optimizer)
        self.rho = rho
        self.adaptive = adaptive
        self.use_gc = use_gc
        self.perturb_eps = perturb_eps
    
    def reset(self):
        self.old_p = []
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.old_p[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.old_p[self._get_variable_index(var)])

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.old_p = []
        for var in var_list:
            self.old_p.append(tf.Variable(var))
            self._track_variable(self.old_p[-1])
    
    def first_step(self, grads, trainable_variables):
        grad_norm = self.grad_norm(grads, trainable_variables) + self.perturb_eps
        scale = self.rho / grad_norm

        for p, g in zip(trainable_variables, grads):
            if self.use_gc:
                size = len(g.shape)
                if size > 1:
                    grads[self._get_variable_index(p)] += tf.reduce_mean(-g, axis=tuple(range(1, size)), keepdims=True)

            self.old_p[self._get_variable_index(p)].assign(p)
            e_w = (tf.pow(p, 2) if self.adaptive else 1.0) * g * tf.cast(scale, p.dtype)

            p.assign_add(e_w)
    
    def second_step(self, grads, trainable_variables):
        for p, g in zip(trainable_variables, grads):
            p.assign(self.old_p[self._get_variable_index(p)])

        if self.tape is None and self.loss is None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables))
        elif self.tape is not None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.tape)
        else:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.loss)
    
    def apply_gradients(self, grads_and_vars, tape=None, loss=None):
        self.tape = tape
        self.loss = loss
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

    def update_step(self, grads, trainable_variables, learning_rate):
        self.first_step(grads, trainable_variables)

        self.second_step(grads, trainable_variables)
    
    def grad_norm(self, grads, trainable_variables):
        norms = []
        for p, g in zip(trainable_variables, grads):
            scale = tf.abs(p) if self.adaptive else 1.0
            norm_val = tf.norm(scale * g, ord=2)
            norms.append(norm_val)
        total_norm = tf.norm(tf.stack(norms), ord=2)
        return total_norm
    
    def state_dict(self):
        state_dict1 = dict()
        state_dict2 = dict()
        return self.save_own_variables(state_dict1), self._base_optimizer.save_own_variables(state_dict2)
    
    def load_state_dict(self, state_dict):
        self.load_own_variables(state_dict[0])
        self._base_optimizer.load_own_variables(state_dict[1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_optimizer_": tf.keras.optimizers.serialize(self.base_optimizer),
                "rho": self.rho,
                "adaptive": self.adaptive,
                "use_gc": self.use_gc,
                "perturb_eps": self.perturb_eps,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class GSAM(optimizer.Optimizer):
    def __init__(
        self,
        model,
        base_optimizer,
        rho_scheduler,
        alpha=0.4,
        adaptive=False,
        perturb_eps=1e-12,
        name="gsam",
        **kwargs,
    ):
        super().__init__(learning_rate=1.,name=name)
        self.base_optimizer = base_optimizer
        self.base_optimizer_ = tf.keras.optimizers.serialize(base_optimizer)
        self.rho_scheduler = rho_scheduler
        self.alpha = alpha
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.rho_t = 0.0
        
        if hasattr(tf.distribute.ReduceOp, 'MEAN'):
            self.grad_reduce = tf.distribute.ReduceOp.MEAN
            self.manual_average = False
        else:
            self.grad_reduce = tf.distribute.ReduceOp.SUM
            self.manual_average = True
        
        self.update_rho_t()
    
    def reset(self):
        self.old_g = []
        self.e_w = []
        self.sharpness = []
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.old_g[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.old_g[self._get_variable_index(var)])
            self.e_w[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.e_w[self._get_variable_index(var)])
            self.sharpness[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.sharpness[self._get_variable_index(var)])
    
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t
    
    def perturb_weights(self, rho, grads, trainable_variables):
        grad_norm = self.grad_norm(weight_adaptive=self.adaptive)
        scale = rho / (grad_norm + self.perturb_eps)

        for p, g in zip(trainable_variables, grads):
            self.old_g[self._get_variable_index(p)].assign(g)

            e_w = (tf.pow(p, 2) if self.adaptive else 1.0) * g * tf.cast(scale, p.dtype)

            p.assign_add(e_w)

            self.e_w[self._get_variable_index(p)].assign(e_w)
    
    def un_perturb(self):
        for p in self._trainable_variables:
            p.assign_sub(self.e_w[self._get_variable_index(p)])
    
    def gradient_decompose(self, grads, trainable_variables, alpha = 0.0):
        inner_prod = 0.0
        for p, g in zip(trainable_variables, grads):
            inner_prod += tf.reduce_sum(self.old_g[self._get_variable_index(p)] * g)

        new_grad_norm = self.grad_norm(by=None)
        old_grad_norm = self.grad_norm(by='old_g')

        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        for p, g in zip(trainable_variables, grads):
            vertical = self.old_g[self._get_variable_index(p)] - cosine * old_grad_norm * g / (
                new_grad_norm + self.perturb_eps
            )
            grads[self._get_variable_index(p)] += vertical * -alpha
    
    def sync_grad(self, grads, trainable_variables):
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            for p, g in zip(trainable_variables, grads):
                grads[self._get_variable_index(p)] = strategy.reduce(self.grad_reduce, g, axis=None)
                if self.manual_average:
                    grads[self._get_variable_index(p)] /= float(strategy.num_replicas_in_sync)
    
    def maybe_no_sync(self):
        return self.model.no_sync() if tf.distribute.has_strategy() else ExitStack()

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.old_g = []
        self.e_w = []
        for var in var_list:
            self.old_g.append(tf.Variable(var))
            self._track_variable(self.old_g[-1])
            self.e_w.append(tf.Variable(var))
            self._track_variable(self.e_w[-1])
    
    def apply_gradients(self, grads_and_vars, tape=None, loss=None):
        self.tape = tape
        self.loss = loss
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

    def update_step(self, grads, trainable_variables, learning_rate):
        with self.maybe_no_sync():
            self.perturb_weights(rho=self.rho_t)

            if hasattr(self.model, 'layer_list'):
                for layer in self.model.layer_list:
                    if isinstance(layer, nn.batch_norm):
                        layer.backup_momentum = layer.momentum
                        layer.momentum = 0
            else:
                for layer in self.model.layers:
                    if isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.backup_momentum = layer.momentum
                        layer.momentum = 0.0
                    if hasattr(layer, 'layers'):
                        for layer in layer.layers:
                            if isinstance(layer, tf.keras.layers.BatchNormalization):
                                layer.backup_momentum = layer.momentum
                                layer.momentum = 0.0

            self.gradient_decompose(self.alpha)

            self.un_perturb()

        self.sync_grad()

        if self.tape is None and self.loss is None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables))
        elif self.tape is not None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.tape)
        else:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.loss)

        if hasattr(self.model, 'layer_list'):
            for layer in self.model.layer_list:
                if isinstance(layer, nn.batch_norm) and hasattr(layer, 'backup_momentum'):
                    layer.momentum = layer.backup_momentum
        else:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.momentum = layer.backup_momentum
                if hasattr(layer, 'layers'):
                    for layer in layer.layers:
                        if isinstance(layer, tf.keras.layers.BatchNormalization):
                            layer.momentum = layer.backup_momentum
    
    def grad_norm(self, grads, trainable_variables):
        norms = []
        for p, g in zip(trainable_variables, grads):
            scale = tf.abs(p) if self.adaptive else 1.0
            norm_val = tf.norm(scale * g, ord=2)
            norms.append(norm_val)
        total_norm = tf.norm(tf.stack(norms), ord=2)
        return total_norm
    
    def state_dict(self):
        state_dict1 = dict()
        state_dict2 = dict()
        return self.save_own_variables(state_dict1), self._base_optimizer.save_own_variables(state_dict2)
    
    def load_state_dict(self, state_dict):
        self.load_own_variables(state_dict[0])
        self._base_optimizer.load_own_variables(state_dict[1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_optimizer_": tf.keras.optimizers.serialize(self.base_optimizer),
                "rho_scheduler": self.rho_scheduler,
                "alpha": self.alpha,
                "adaptive": self.adaptive,
                "perturb_eps": self.perturb_eps,
                "rho_t": self.rho_t,
                "grad_reduce": self.grad_reduce,
                "manual_average": self.manual_average,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class WSAM(optimizer.Optimizer):
    def __init__(
        self,
        model,
        base_optimizer,
        rho=0.05,
        gamma=0.9,
        adaptive=False,
        decouple=True,
        max_norm=None,
        eps=1e-12,
        name="wsam",
        **kwargs,
    ):
        super().__init__(learning_rate=1.,name=name)
        self.base_optimizer = base_optimizer
        self.base_optimizer_ = tf.keras.optimizers.serialize(base_optimizer)
        self.rho = rho
        self.gamma = gamma
        self.adaptive = adaptive
        self.decouple = decouple
        self.max_norm = max_norm
        self.sam_eps = eps
        self.alpha = gamma / (1.0 - gamma)
    
    def reset(self):
        self.e_w = []
        self.grad = []
        self.sharpness = []
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.e_w[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.e_w[self._get_variable_index(var)])
            self.grad[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.grad[self._get_variable_index(var)])
            self.sharpness[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.sharpness[self._get_variable_index(var)])

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.e_w = []
        self.grad = []
        self.sharpness = []
        for var in var_list:
            self.e_w.append(tf.Variable(var))
            self._track_variable(self.e_w[-1])
            self.grad.append(tf.Variable(var))
            self._track_variable(self.grad[-1])
            self.sharpness.append(tf.Variable(var))
            self._track_variable(self.sharpness[-1])
    
    def first_step(self, grads, trainable_variables):
        grad_norm = self.grad_norm(grads, trainable_variables)
        scale = self.rho / (grad_norm + self.sam_eps)

        for p, g in zip(trainable_variables, grads):
            e_w = (tf.pow(p, 2) if self.adaptive else 1.0) * g * tf.cast(scale, p.dtype)

            p.assign_add(e_w)
            
            self.e_w[self._get_variable_index(p)].assign(e_w)
            
            if tf.distribute.has_strategy():  # pragma: no cover
                strategy = tf.distribute.get_strategy()
                grads[self._get_variable_index(p)] = strategy.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None)
            
        for p, g in zip(trainable_variables, grads):
            self.grad[self._get_variable_index(p)].assign(g)
    
    def second_step(self, grads, trainable_variables):
        for p, g in zip(trainable_variables, grads):
            if tf.distribute.has_strategy():  # pragma: no cover
                strategy = tf.distribute.get_strategy()
                grads[self._get_variable_index(p)] = strategy.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None)

            p.assign_add(self.e_w[self._get_variable_index(p)] * -1.0)

        if self.max_norm is not None:
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_norm)

        for p, g in zip(trainable_variables, clipped_grads):
            if not self.decouple:
                grads[self._get_variable_index(p)] = g * self.alpha + self.grad[self._get_variable_index(p)] * (1.0 - self.alpha)
            else:
                self.sharpness[self._get_variable_index(p)].assign(g - self.grad[self._get_variable_index(p)])
                grads[self._get_variable_index(p)] = g * 0.0 + self.grad[self._get_variable_index(p)] * 1.0

        if self.tape is None and self.loss is None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables))
        elif self.tape is not None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.tape)
        else:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.loss)

        if self.decouple:
            for p in zip(trainable_variables):
                lr = tf.cast(self.base_optimizer._learning_rate, p.dtype)
                p.assign_add(self.sharpness[self._get_variable_index(p)] * -lr * self.alpha)
    
    def apply_gradients(self, grads_and_vars, tape=None, loss=None):
        self.tape = tape
        self.loss = loss
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

    def update_step(self, grads, trainable_variables, learning_rate):
        if hasattr(self.model, 'layer_list'):
            for layer in self.model.layer_list:
                if isinstance(layer, nn.batch_norm) and hasattr(layer, 'backup_momentum'):
                    layer.momentum = layer.backup_momentum
        else:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.momentum = layer.backup_momentum
                if hasattr(layer, 'layers'):
                    for layer in layer.layers:
                        if isinstance(layer, tf.keras.layers.BatchNormalization):
                            layer.momentum = layer.backup_momentum

        self.first_step(grads, trainable_variables)

        if hasattr(self.model, 'layer_list'):
            for layer in self.model.layer_list:
                if isinstance(layer, nn.batch_norm):
                    layer.backup_momentum = layer.momentum
                    layer.momentum = 0
        else:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.backup_momentum = layer.momentum
                    layer.momentum = 0.0
                if hasattr(layer, 'layers'):
                    for layer in layer.layers:
                        if isinstance(layer, tf.keras.layers.BatchNormalization):
                            layer.backup_momentum = layer.momentum
                            layer.momentum = 0.0

        self.second_step(grads, trainable_variables)
    
    def grad_norm(self, grads, trainable_variables):
        norms = []
        for p, g in zip(trainable_variables, grads):
            scale = tf.abs(p) if self.adaptive else 1.0
            norm_val = tf.norm(scale * g, ord=2)
            norms.append(norm_val)
        total_norm = tf.norm(tf.stack(norms), ord=2)
        return total_norm
    
    def state_dict(self):
        state_dict1 = dict()
        state_dict2 = dict()
        return self.save_own_variables(state_dict1), self._base_optimizer.save_own_variables(state_dict2)
    
    def load_state_dict(self, state_dict):
        self.load_own_variables(state_dict[0])
        self._base_optimizer.load_own_variables(state_dict[1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_optimizer_": tf.keras.optimizers.serialize(self.base_optimizer),
                "rho": self.rho,
                "gamma": self.gamma,
                "adaptive": self.adaptive,
                "decouple": self.decouple,
                "max_norm": self.max_norm,
                "sam_eps": self.sam_eps,
                "alpha": self.alpha,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class BSAM(optimizer.Optimizer):
    def __init__(
        self,
        num_data,
        lr=5e-1,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4,
        rho=0.05,
        adaptive=False,
        damping=0.1,
        name="bsam",
        **kwargs,
    ):
        super().__init__(learning_rate=1., weight_decay=weight_decay, name=name)
        self.num_data = num_data
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.adaptive = adaptive
        self.damping = damping
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.s[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, initializer="ones", name="s"
                                                    )
            self.noisy_gradient[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="noisy_gradient"
                                                    )
            self.momentum[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="momentum"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.s = []
        self.noisy_gradient = []
        self.momentum = []
        for var in var_list:
            self.s.append(
                self.add_variable_from_reference(
                    reference_variable=var, initializer="ones", name="s"
                )
            )
            self.noisy_gradient.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="noisy_gradient"
                )
            )
            self.momentum.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )
    
    def first_step(self, grads, trainable_variables):
        for p, g in zip(trainable_variables, grads):
            noise = tf.random.normal(
                        shape=p.shape,
                        mean=0.0,
                        stddev=1.0 / (self.num_data * self.s[self._get_variable_index(p)])
                    )

            p.assign_add(noise)

    def second_step(self, grads, trainable_variables):
        for p, g in zip(trainable_variables, grads):
            self.noisy_gradient[self._get_variable_index(p)].assign(g)

            e_w = (tf.pow(p, 2) if self.adaptive else 1.0) * self.rho * g / self.s[self._get_variable_index(p)]

            p.assign_add(e_w)

    def third_step(self, grads, trainable_variables):
        for p, g in zip(trainable_variables, grads):
            momentum = self.momentum[self._get_variable_index(p)]
            s = self.s[self._get_variable_index(p)]
            momentum.assign(momentum * self.beta1 + g * self.weight_decay * (1.0 - self.beta1))

            var = tf.pow((tf.sqrt(s) * tf.abs(g) + (self.weight_decay + self.damping)), 2)
            s.assign(s * self.beta2 + var * (1.0 - self.beta2))

            p.assign_add(momentum / s * -self.lr)
    
    def apply_gradients(self, grads_and_vars):
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

    def update_step(self, grads, trainable_variables, learning_rate):
        self.first_step(grads, trainable_variables)

        self.second_step(grads, trainable_variables)

        self.third_step(grads, trainable_variables)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_data": self.num_data,
                "lr": self.lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "rho": self.rho,
                "adaptive": self.adaptive,
                "damping": self.damping,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class LookSAM(optimizer.Optimizer):
    def __init__(
        self,
        base_optimizer,
        rho=0.1,
        k=10,
        alpha=0.7,
        adaptive=False,
        use_gc=False,
        perturb_eps=1e-12,
        step=True,
        name="looksam",
        **kwargs,
    ):
        super().__init__(learning_rate=1.,name=name)
        self.base_optimizer = base_optimizer
        self.base_optimizer_ = tf.keras.optimizers.serialize(base_optimizer)
        self.rho = rho
        self.k = k
        self.alpha = alpha
        self.adaptive = adaptive
        self.use_gc = use_gc
        self.perturb_eps = perturb_eps
        self.step = step
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.old_p[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.old_p[self._get_variable_index(var)])
            self.old_grad_p[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.old_grad_p[self._get_variable_index(var)])
            self.gv[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.gv[self._get_variable_index(var)])

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.old_p = []
        self.old_grad_p = []
        self.gv = []
        for var in var_list:
            self.old_p.append(tf.Variable(var))
            self._track_variable(self.old_p[-1])
            self.old_grad_p.append(tf.Variable(var))
            self._track_variable(self.old_grad_p[-1])
            self.gv.append(tf.Variable(var))
            self._track_variable(self.gv[-1])
    
    def get_step(self):
        return self.base_optimizer.iterations if self.step else 0
    
    def first_step(self, grads, trainable_variables):
        def true_fn():
            pass
        def false_fn():
            grad_norm = self.grad_norm() + self.perturb_eps
            scale = self.rho / grad_norm
    
            for p, g in zip(trainable_variables, grads):
                if self.use_gc:
                    size = len(g.shape)
                    if size > 1:
                        grads[self._get_variable_index(p)] += tf.reduce_mean(-g, axis=tuple(range(1, size)), keepdims=True)
    
                self.old_p[self._get_variable_index(p)].assign(p)
                self.old_grad_p[self._get_variable_index(p)].assign(g)
    
                e_w = (tf.pow(p, 2) if self.adaptive else 1.0) * g * tf.cast(scale, p.dtype)
                p.assign_add(e_w)
        tf.cond(self.get_step() % self.k != 0, true_fn, false_fn)

    def second_step(self, grads, trainable_variables):
        step = self.get_step()

        for p, g in zip(trainable_variables, grads):
            grad_norm = tf.norm(g, ord=2)
            
            def true_fn():
                old_grad_p = self.old_grad_p[self._get_variable_index(p)]

                g_grad_norm = old_grad_p / tf.norm(old_grad_p, ord=2)
                g_s_grad_norm = g / grad_norm

                self.gv[self._get_variable_index(p)].assign(
                    g - grad_norm * tf.reduce_sum(g_grad_norm * g_s_grad_norm) * g_grad_norm
                )
            
            def false_fn():
                gv = self.gv[self._get_variable_index(p)]
                grads[self._get_variable_index(p)] += grad_norm / (tf.norm(gv, ord=2) + 1e-8) * gv * self.alpha
            
            tf.cond(step % self.k == 0, true_fn, false_fn)

            p.assign(self.old_p[self._get_variable_index(p)])

        if self.tape is None and self.loss is None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables))
        elif self.tape is not None:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.tape)
        else:
            self.base_optimizer.apply_gradients(zip(grads, trainable_variables), self.loss)
    
    def apply_gradients(self, grads_and_vars, tape=None, loss=None):
        self.tape = tape
        self.loss = loss
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

    def update_step(self, grads, trainable_variables, learning_rate):
        self.first_step(grads, trainable_variables)

        self.second_step(grads, trainable_variables)
    
    def grad_norm(self, grads, trainable_variables):
        norms = []
        for p, g in zip(trainable_variables, grads):
            scale = tf.abs(p) if self.adaptive else 1.0
            norm_val = tf.norm(scale * g, ord=2)
            norms.append(norm_val)
        total_norm = tf.norm(tf.stack(norms), ord=2)
        return total_norm
    
    def state_dict(self):
        state_dict1 = dict()
        state_dict2 = dict()
        return self.save_own_variables(state_dict1), self._base_optimizer.save_own_variables(state_dict2)
    
    def load_state_dict(self, state_dict):
        self.load_own_variables(state_dict[0])
        self._base_optimizer.load_own_variables(state_dict[1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_optimizer_": tf.keras.optimizers.serialize(self.base_optimizer),
                "rho": self.rho,
                "k": self.k,
                "alpha": self.alpha,
                "adaptive": self.adaptive,
                "use_gc": self.use_gc,
                "perturb_eps": self.perturb_eps,
                "step": self.step,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
