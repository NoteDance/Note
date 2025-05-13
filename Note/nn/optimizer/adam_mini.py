""" AdamMini
https://arxiv.org/abs/2406.16793

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class AdamMini(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        betas=(0.9,0.999),
        epsilon=1e-8,
        weight_decay=0.1,
        model_sharding=False,
        num_embeds=2048,
        num_heads=32,
        num_query_groups=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adammini",
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
        self.model_sharding = model_sharding
        self.num_embeds = num_embeds
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups if num_query_groups is not None else num_embeds
        self.embed_blocks = {'embed', 'embd', 'wte', 'lm_head.weight', 'output.weight'}
        self.qk_blocks = {'k_proj.weight', 'q_proj.weight', 'wq.weight', 'wk.weight'}
        self.world_size = tf.distribute.get_strategy().num_replicas_in_sync
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            var = tf.Variable(tf.cast(var, tf.float32))
            if any(block in self.names for block in self.embed_blocks):
                self.m[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                            reference_variable=var, name="m"
                                                        )
                self.v[self._get_variable_index(var)] = self.add_variable_from_reference(
                                                            reference_variable=var, name="v"
                                                        )
            elif any(block in self.names for block in self.qk_blocks):
                self.m[self._get_variable_index(var)] = self.add_variable_from_reference(
                        reference_variable=tf.Variable(tf.reshape(var, (-1, self.parameter_per_head))), name="m"
                    )
                
                self.v_mean.assign(tf.zeros(self.m[-1].shape[0]))
            elif 'attn.attn.weight' in self.names or 'attn.qkv.weight' in self.names:
                self.m[self._get_variable_index(var)] = self.add_variable_from_reference(
                        reference_variable=tf.Variable(tf.reshape(var, (self.n_head, self.q_per_kv + 2, -1))), name="m"
                    )
                self.v_mean.assign(tf.zeros((self.n_head, self.q_per_kv + 2)))
            else:
                self.m[self._get_variable_index(var)] = self.add_variable_from_reference(
                        reference_variable=var, name="m"
                    )
                self.v_mean.assign(0)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.weight_decay_list = []
        self.head = []
        self.v_mean = []
        self.m = []
        self.v = []
        
        self.names = []
        for var in var_list:
            name = var.name
            self.names.append(name)
        for var in var_list:
            var = tf.Variable(tf.cast(var, tf.float32))
            if ('norm' in name or 'ln_f' in name):
                self.weight_decay_list.append(0.0)
            else:
                self.weight_decay_list.append(self.weight_decay)
                
            if any(block in self.names for block in self.embed_blocks):
                self.m.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="m"
                    )
                )
                self.v.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="v"
                    )
                )
            elif any(block in self.names for block in self.qk_blocks):
                self.parameter_per_head = self.num_embeds * self.num_embeds // self.num_heads
                self.m.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.Variable(tf.reshape(var, (-1, self.parameter_per_head))), name="m"
                    )
                )
                self.head.append(self.m[-1].shape[0])
                self.v_mean.append(tf.Variable(tf.zeros(self.m[-1].shape[0])))
                self._track_variable(self.v_mean[-1])
            elif 'attn.attn.weight' in self.names or 'attn.qkv.weight' in self.names:
                self.n_head = self.num_heads
                self.q_per_kv = self.num_embeds // self.num_query_groups
                self.m.append(
                    self.add_variable_from_reference(
                        reference_variable=tf.Variable(tf.reshape(var, (self.n_head, self.q_per_kv + 2, -1))), name="m"
                    )
                )
                self.v_mean.append(tf.Variable(tf.zeros((self.n_head, self.q_per_kv + 2))))
                self._track_variable(self.v_mean[-1])
            else:
                dim = tf.size(var, out_type=tf.float32)

                reduced = False
                if self.model_sharding and self.world_size > 1:
                    replica_ctx = tf.distribute.get_replica_context()
                    dims = replica_ctx.all_gather(dim, axis=0)
                    positive_count = tf.reduce_sum(tf.cast(dims > 0, tf.int32))
                    dim = tf.reduce_sum(dims)
                    reduced = tf.cast(positive_count, tf.int32) >= 2

                self.m.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="m"
                    )
                )
                self.v_mean.append(tf.Variable(tf.zeros(())))
                self._track_variable(self.v_mean[-1])
                self.dimension = dim
                self.reduced = reduced
    
    def step_embed(
        self,
        p,
        grad,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2_sq,
        eps,
    ):
        m, v = self.m[self._get_variable_index(p)], self.v[self._get_variable_index(p)]

        m.assign_add((grad - m) * (1.0 - beta1))
        v.assign(v * beta2 + grad * tf.math.conj(grad) * (1.0 - beta2))

        h = (tf.sqrt(v) / bias_correction2_sq) + eps

        if p.dtype != tf.float32:
            p.assign_add(tf.cast(-lr / bias_correction1 * m / h, p.dtype))
        else:
            p.assign_add(-lr / bias_correction1 * m / h)
    
    def step_attn_proj(
        self,
        p,
        grad,
        parameter_per_head,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2_sq,
        eps,
    ):
        m, v = self.m[self._get_variable_index(p)], self.v_mean[self._get_variable_index(p)]

        head = self.head[self._get_variable_index(p)]
        grad = tf.reshape(grad, (head, parameter_per_head))

        m.assign_add((grad - m) * (1.0 - beta1))

        tmp_lr = tf.reduce_mean(grad * grad, axis=1)
        v.assign(v * beta2 + tmp_lr * (1.0 - beta2))

        h = (tf.sqrt(v) / bias_correction2_sq) + eps

        update = tf.reshape((1 / (h * bias_correction1)), (head, 1)) * m

        if len(p.shape) > 1:
            d0, d1 = p.shape
            update = tf.reshape(update, (d0, d1))
        else:
            update = tf.reshape(update, (-1))

        if p.dtype != tf.float32:
            p.assign_add(tf.cast(update * -lr, p.dtype))
        else:
            p.assign_add(update * -lr)
    
    def step_attn(
        self,
        p,
        grad,
        num_heads,
        q_per_kv,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2_sq,
        eps,
    ):
        m, v = self.m[self._get_variable_index(p)], self.v_mean[self._get_variable_index(p)]

        grad = tf.reshape(grad, (num_heads, q_per_kv + 2, -1))

        m.assign_add((grad - m) * (1.0 - beta1))

        tmp_lr = tf.reduce_mean(grad * grad, axis=2)
        v.assign(v * beta2 + tmp_lr * (1.0 - beta2))

        h = (tf.sqrt(v) / bias_correction2_sq) + eps

        update = tf.reshape((1 / (h * bias_correction1)), (num_heads, q_per_kv + 2, -1)) * m

        if len(p.shape) > 1:
            d0, d1 = p.shape
            update = tf.reshape(update, (d0, d1))
        else:
            update = tf.reshape(update, (-1))

        if p.dtype != tf.float32:
            p.assign_add(tf.cast(update * -lr, p.dtype))
        else:
            p.assign_add(update * -lr)
    
    def step_lefts(
        self,
        p,
        grad,
        lr,
        beta1,
        beta2,
        bias_correction1,
        bias_correction2_sq,
        eps,
    ):
        tmp_lr = tf.reduce_sum(grad * grad)
        
        def true_fn(tmp_lr = tmp_lr):
            replica_ctx = tf.distribute.get_replica_context()
            tmp_lr = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, tmp_lr)
            return tmp_lr
        def false_fn():
            return tmp_lr
        tmp_lr = tf.cond(self.reduced, true_fn, false_fn)

        tmp_lr /= self.dimension

        m, v = self.m[self._get_variable_index(p)], self.v_mean[self._get_variable_index(p)]

        m.assign_add((grad - m) * (1.0 - beta1))
        v.assign(v * beta2 + tmp_lr * (1.0 - beta2))

        h = (tf.sqrt(v) / bias_correction2_sq) + eps

        stepsize = (1 / bias_correction1) / h

        update = m * stepsize

        if p.dtype != tf.float32:
            p.assign_add(tf.cast(update * -lr, p.dtype))
        else:
            p.assign_add(update * -lr)

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'AdamMini does not support sparse gradients')
        
        lr = tf.cast(learning_rate, variable.dtype)
        
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        beta1, beta2 = self.betas
        
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        bias_correction2_sq = tf.sqrt(bias_correction2)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
        
        if any(block in self.names for block in self.embed_blocks):
            self.step_embed(
                variable, gradient, lr, beta1, beta2, bias_correction1, bias_correction2_sq, self.epsilon
            )
        elif any(block in self.names for block in self.qk_blocks):
            self.step_attn_proj(
                variable,
                gradient,
                self.parameter_per_head,
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2_sq,
                self.epsilon,
            )
        elif 'attn.attn.weight' in self.names or 'attn.qkv.weight' in self.names:
            self.step_attn(
                variable,
                gradient,
                self.n_head,
                self.q_per_kv,
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2_sq,
                self.epsilon,
            )
        else:
            self.step_lefts(
                variable,
                gradient,
                lr,
                beta1,
                beta2,
                bias_correction1,
                bias_correction2_sq,
                self.epsilon,
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "betas": self.betas,
                "epsilon": self.epsilon,
                "model_sharding": self.model_sharding,
                "num_embeds": self.num_embeds,
                "num_heads": self.num_heads,
                "num_query_groups": self.num_query_groups,
                "embed_blocks": self.embed_blocks,
                "qk_blocks": self.qk_blocks,
                "world_size": self.world_size,
                "head": self.head,
                "parameter_per_head": self.parameter_per_head,
                "n_head": self.n_head,
                "q_per_kv": self.q_per_kv,
                "dimension": self.dimension,
                "reduced": self.reduced,
                "names": self.names,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
