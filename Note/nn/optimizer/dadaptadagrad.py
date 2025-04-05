""" DAdaptAdaGrad
https://arxiv.org/abs/2301.07733

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


def sparse_mask(dense_tensor, mask_sparse):
    indices = mask_sparse.indices  # [N, ndims]
    values = tf.gather_nd(dense_tensor, indices)
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=mask_sparse.dense_shape)


class DAdaptAdaGrad(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1.0,
        epsilon=0.0,
        weight_decay=0.0,
        momentum=0.0,
        d0=1e-6,
        growth_rate=float('inf'),
        weight_decouple=True,
        fixed_decay=False,
        bias_correction=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="dadaptadagrad",
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
        self.epsilon = epsilon
        self.momentum = momentum
        self.d0_ = d0
        self.growth_rate = growth_rate
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.bias_correction = bias_correction
    
    def reset(self):
        self.gsq_weighted = tf.Variable(0.0)
        self.sk_sq_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.g_sq = tf.Variable(0.0)
        self.sk_sq_weighted_change = tf.Variable(0.0)
        self.sk_l1_change = tf.Variable(0.0)
        self.d0 = tf.Variable(self.d0_)
        self._track_variable(self.gsq_weighted)
        self._track_variable(self.sk_sq_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.g_sq)
        self._track_variable(self.sk_sq_weighted_change)
        self._track_variable(self.sk_l1_change)
        self._track_variable(self.d0)
        for var in self._trainable_variables:
            self.alpha_k[self._get_variable_index(var)] =  tf.Variable(tf.ones_like(var) * 1e-6)
            self._track_variable(self.alpha_k[self._get_variable_index(var)])
            self.sk[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="sk"
                                                    )
            self.x0[self._get_variable_index(var)] = tf.Variable(var)
            self._track_variable(self.x0[self._get_variable_index(var)])
            if tf.keras.backend.is_sparse(var):
                self.weighted_sk[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                            reference_variable=var, name="weighted_sk"
                                                        )
            else:
                self.weighted_sk.append(None)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.alpha_k = []
        self.sk = []
        self.x0 = []
        self.weighted_sk = []
        self.gsq_weighted = tf.Variable(0.0)
        self.sk_sq_weighted = tf.Variable(0.0)
        self.sk_l1 = tf.Variable(0.0)
        self.g_sq = tf.Variable(0.0)
        self.sk_sq_weighted_change = tf.Variable(0.0)
        self.sk_l1_change = tf.Variable(0.0)
        self.d0 = tf.Variable(self.d0_)
        self._track_variable(self.gsq_weighted)
        self._track_variable(self.sk_sq_weighted)
        self._track_variable(self.sk_l1)
        self._track_variable(self.g_sq)
        self._track_variable(self.sk_sq_weighted_change)
        self._track_variable(self.sk_l1_change)
        self._track_variable(self.d0)
        for var in var_list:
            self.alpha_k.append(tf.Variable(tf.ones_like(var) * 1e-6))
            self._track_variable(self.alpha_k[-1])
            self.sk.append(self.add_variable_from_reference(
                                reference_variable=var, name="sk"
                                                    ))
            self.x0.append(tf.Variable(var))
            self._track_variable(self.x0[-1])
            if tf.keras.backend.is_sparse(var):
                self.weighted_sk.append(self.add_variable_from_reference(
                                    reference_variable=var, name="weighted_sk"
                                                        ))
            else:
                self.weighted_sk.append(None)
        
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        d_lr = self.d0 * self.lr
        
        for var, grad in zip(trainable_variables, grads):
            sk = self.sk[self._get_variable_index(var)]
            alpha_k = self.alpha_k[self._get_variable_index(var)]
            
            if tf.keras.backend.is_sparse(grad):
                weighted_sk = self.weighted_sk[self._get_variable_index(var)]
                
                grad = tf.sparse.reorder(grad)
                
                vk = tf.pow(grad.values, 2)
                
                sk_masked = sparse_mask(sk, grad)
                old_sk_l1_masked = tf.reduce_sum(tf.abs(sk_masked.values))
                
                grad_scaled = tf.sparse.SparseTensor(indices=grad.indices, 
                                                     values=grad.values * d_lr,
                                                     dense_shape=grad.dense_shape)
                sk = tf.sparse.add(sk, grad_scaled)
                
                sk_masked = sparse_mask(sk, grad)
                alpha_k_masked = sparse_mask(alpha_k, grad)
                weighted_sk_masked = sparse_mask(weighted_sk, grad)
                
                alpha_k_p1_masked = alpha_k_masked.values + vk
                alpha_k_delta_masked = alpha_k_p1_masked - alpha_k_masked.values
                alpha_k_delta = tf.sparse.SparseTensor(indices=grad.indices, 
                                                       values=alpha_k_delta_masked, 
                                                       dense_shape=grad.dense_shape)
                alpha_k = tf.sparse.add(alpha_k, alpha_k_delta)
                
                de_nom = tf.sqrt(alpha_k_p1_masked + self.epsilon)
                
                grad_sq = tf.reduce_sum(vk / de_nom)
                self.g_sq.assign_add(grad_sq)
                
                weighted_sk_p1_masked = tf.pow(sk_masked.values, 2) / de_nom
                delta_weighted_sk = tf.reduce_sum(weighted_sk_p1_masked) - tf.reduce_sum(weighted_sk_masked.values)
                self.sk_sq_weighted_change.assign_add(delta_weighted_sk)
                
                weighted_sk_p1_delta_masked = weighted_sk_p1_masked - weighted_sk_masked.values
                weighted_sk_p1_delta = tf.sparse.SparseTensor(indices=grad.indices,
                                                              values=weighted_sk_p1_delta_masked,
                                                              dense_shape=grad.dense_shape)
                weighted_sk = tf.sparse.add(weighted_sk, weighted_sk_p1_delta)
                
                sk_l1_masked = tf.reduce_sum(tf.abs(sk_masked.values))
                self.sk_l1_change.assign_add(sk_l1_masked - old_sk_l1_masked)
            else:
                if self.weight_decouple:
                    var.assign(var * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else self.lr)))
                elif self.weight_decay > 0.0:
                    grad += var * self.weight_decay
                
                old_sk_sq_weighted_param = tf.reduce_sum(tf.pow(sk, 2) / (tf.sqrt(alpha_k) + self.epsilon))
                old_sk_l1_param = tf.reduce_sum(tf.abs(sk))
                
                alpha_k.assign_add(tf.pow(grad, 2))
                grad_sq = tf.reduce_sum(tf.pow(grad, 2) / (tf.sqrt(alpha_k) + self.epsilon))
                self.g_sq.assign_add(grad_sq)
                
                d_lr = tf.cast(d_lr, dtype=var.dtype)
                
                sk.assign_add(grad * d_lr)
                
                sk_sq_weighted_param = tf.reduce_sum(tf.pow(sk, 2) / (tf.sqrt(alpha_k) + self.epsilon))
                sk_l1_param = tf.reduce_sum(tf.abs(sk))
                
                self.sk_sq_weighted_change.assign_add(sk_sq_weighted_param - old_sk_sq_weighted_param)
                self.sk_l1_change.assign_add(sk_l1_param - old_sk_l1_param)
            
        self.sk_sq_weighted.assign_add(self.sk_sq_weighted_change)
        self.gsq_weighted.assign_add(self.g_sq * d_lr ** 2)  # fmt: skip
        self.sk_l1.assign_add(self.sk_l1_change)
        
        def update_fn():
            if self.lr > 0.0:
                d_hat = (self.sk_sq_weighted - self.gsq_weighted) / self.sk_l1
                d = tf.maximum(self.d0, tf.minimum(d_hat, self.d0 * self.growth_rate))
            
            self.d0.assign(d)
            for var, grad in zip(trainable_variables, grads):
                alpha_k = self.alpha_k[self._get_variable_index(var)]
                sk = self.sk[self._get_variable_index(var)]
                x0 = self.x0[self._get_variable_index(var)]
                
                if tf.keras.backend.is_sparse(grad):
                    grad = tf.sparse.reorder(grad)
                    
                    sk_masked = sparse_mask(sk, grad).values
                    alpha_k_masked = sparse_mask(alpha_k, grad).values
                    x0_masked = sparse_mask(x0, grad).values
                    p_masked = sparse_mask(var, grad).values
                
                    loc_masked = x0_masked - sk_masked / tf.sqrt(alpha_k_masked + self.epsilon)
                    loc_delta_masked = loc_masked - p_masked
                    loc_delta = tf.sparse.SparseTensor(indices=grad.indices, 
                                                       values=loc_delta_masked, 
                                                       dense_shape=grad.dense_shape)
                    var.assign_add(tf.sparse.to_dense(loc_delta))
                else:
                    z = x0 - sk / (tf.sqrt(alpha_k) + self.epsilon)
                    
                    if self.momentum > 0.0:
                        var.assign(var * self.momentum + z * (1.0 - self.momentum))
                    else:
                        var.assign(z)
        
        def no_update_fn():
            pass
        
        tf.cond(self.sk_l1 == 0, no_update_fn, update_fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "growth_rate": self.growth_rate,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "bias_correction": self.bias_correction,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass