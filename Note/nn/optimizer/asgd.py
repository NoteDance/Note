import tensorflow as tf
from keras.src.optimizers import optimizer


class ASGD(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-2,
        epsilon=1e-5,
        weight_decay=0.0,
        amplifier=0.02,
        theta=1.0,
        dampening=1.0,
        weight_decouple=True,
        fixed_decay=False,
        maximize=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="asgd",
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
        self.amplifier = amplifier
        self.theta = theta
        self.dampening = dampening
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.maximize = maximize
    
    def reset(self):
        self._iterations.assign(0)
        self.prev_param_norm.assign(0)
        self.prev_grad_norm.assign(0)
        self.curr_param_norm.assign(0)
        self.curr_grad_norm.assign(0)
        self.lr_.assign(self.lr)
        self.theta_.assign(self.theta)

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.prev_param_norm = tf.Variable(tf.zeros((), dtype=tf.float32))
        self.prev_grad_norm = tf.Variable(tf.zeros((), dtype=tf.float32))
        self.curr_param_norm = tf.Variable(tf.zeros((), dtype=tf.float32))
        self.curr_grad_norm = tf.Variable(tf.zeros((), dtype=tf.float32))
        self.lr_ = tf.Variable(tf.cast(self.lr_, tf.float32))
        self.theta_ = tf.Variable(tf.cast(self.theta, tf.float32))
        self._track_variable(self.prev_param_norm)
        self._track_variable(self.prev_grad_norm)
        self._track_variable(self.curr_param_norm)
        self._track_variable(self.curr_grad_norm)
        self._track_variable(self.lr_)
        self._track_variable(self.theta_)

    
    @staticmethod
    def get_norms_by_group(params, grads):
        r"""Get parameter & gradient norm by group."""
        p_norm = tf.zeros((), dtype=tf.float32)
        g_norm = tf.zeros((), dtype=tf.float32)

        for p, g in zip(params, grads):
            p_norm += tf.pow(tf.norm(p), 2)
            g_norm += tf.pow(tf.norm(g), 2)

        p_norm = tf.sqrt(p_norm)
        g_norm = tf.sqrt(g_norm)

        return p_norm, g_norm
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.
    
        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = self.lr_
        theta = self.theta_
        
        p_norm, g_norm = self.get_norms_by_group(trainable_variables, grads)
        
        def true_fn():
            self.prev_param_norm.assign(p_norm)
            self.prev_grad_norm.assign(g_norm)
        def false_fn():
            pass
        tf.cond(self.iterations == 0, true_fn, false_fn)
        self.curr_param_norm.assign(p_norm)
        self.curr_grad_norm.assign(g_norm)
        
        param_diff_norm = self.curr_param_norm - self.prev_param_norm
        grad_diff_norm = self.curr_grad_norm - self.prev_grad_norm
        
        new_lr = lr * tf.sqrt(1 + self.amplifier * theta)
        def true_fn():
            return tf.minimum(new_lr, param_diff_norm / (self.dampening * grad_diff_norm)) + self.epsilon
        def false_fn():
            return new_lr
        new_lr = tf.cond(param_diff_norm > 0 and grad_diff_norm > 0, true_fn, false_fn)
        
        theta.assign(new_lr / lr)
        lr.assign(new_lr)
        
        self.prev_param_norm.assign(self.curr_param_norm)
        self.prev_grad_norm.assign(self.curr_grad_norm)
        for variable, gradient in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(gradient):
                raise RuntimeError(
                    'ASGD does not support sparse gradient.')
            
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                gradient += variable * self.weight_decay
                
            variable.assign_add(gradient * tf.cast(-new_lr, variable.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "epsilon": self.epsilon,
                "amplifier": self.amplifier,
                "theta": self.theta,
                "dampening": self.dampening,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "maximize": self.maximize,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass
