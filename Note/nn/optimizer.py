import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest


class Gradient:
    def __init__(self,lr):
        self.lr=lr
    
    
    def __call__(self,gradient,parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            lr=tf.cast(self.lr, dtype=parameter_flat[i].dtype)
            state_ops.assign(parameter_flat[i],parameter_flat[i]-lr*gradient_flat[i])
        return


class SGD:
    r"""Gradient descent (with momentum) optimizer.

    Update rule for parameter `w` with gradient `g` when `momentum` is 0:

    ```python
    w = w - learning_rate * g
    ```

    Update rule when `momentum` is larger than 0:

    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + velocity
    ```

    When `nesterov=True`, this rule becomes:

    ```python
    velocity = momentum * velocity - learning_rate * g
    w = w + momentum * velocity - learning_rate * g
    ```
    
    Reference:
        - For `nesterov=True`, See [Sutskever et al., 2013](
          http://proceedings.mlr.press/v28/sutskever13.pdf).
    """

    def __init__(
        self,
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = tf.Variable(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")
        self.momentums = []
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        
        if self.flag==0:
            self.flag=1
            for param in parameter_flat:
                self.momentums.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
                
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
            
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, parameter_flat[i].dtype)
            m = None
            momentum = tf.cast(self.momentum, parameter_flat[i].dtype)
            m = self.momentums[i]
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
    
            # TODO(b/204321487): Add nesterov acceleration.
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                add_value = tf.IndexedSlices(
                    -gradient_flat[i].values * lr, gradient_flat[i].indices
                )
                if m is not None:
                    m.assign(m * momentum)
                    m.scatter_add(add_value)
                    if self.nesterov:
                        parameter_flat[i].scatter_add(add_value)
                        parameter_flat[i].assign_add(m * momentum)
                    else:
                        parameter_flat[i].assign_add(m)
                else:
                    parameter_flat[i].scatter_add(add_value)
            else:
                # Dense gradients
                if m is not None:
                    m.assign(-gradient_flat[i] * lr + m * momentum)
                    if self.nesterov:
                        parameter_flat[i].assign_add(-gradient_flat[i] * lr + m * momentum)
                    else:
                        parameter_flat[i].assign_add(m)
                else:
                    parameter_flat[i].assign_add(-gradient_flat[i] * lr)
        return
    
    
class Adagrad:
    r"""Optimizer that implements the Adagrad algorithm.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    Args:
        learning_rate: Initial value for the learning rate:
            either a floating point value,
            or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
            Defaults to 0.001. Note that `Adagrad` tends to benefit from higher
            initial learning rate values compared to other optimizers. To match
            the exact form in the original paper, use 1.0.
        initial_accumulator_value: Floating point value.
            Starting value for the accumulators (per-parameter momentum values).
            Must be non-negative.
        epsilon: Small floating point value used to maintain numerical
            stability.
        {{base_optimizer_keyword_args}}

    Reference:
        - [Duchi et al., 2011](
            http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
    """

    def __init__(
        self,
        lr=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._accumulators = []
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        
        if self.flag==0:
            for param in parameter_flat:
                self._accumulators.append(
                    tf.Variable(
                        tf.fill(param.shape,self.initial_accumulator_value),
                        dtype=param.dtype
                    )
                )
                
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
         
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.learning_rate, parameter_flat[i].dtype)

            accumulator = self._accumulators[i]
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)

            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                accumulator.scatter_add(
                    tf.IndexedSlices(gradient_flat[i].values * gradient_flat[i].values, gradient_flat[i].indices)
                )
                sparse_accumulator = tf.gather(accumulator, indices=gradient_flat[i].indices)
                sparse_denominator = tf.sqrt(sparse_accumulator + self.epsilon)
                parameter_flat[i].scatter_add(
                    tf.IndexedSlices(
                        -lr * gradient_flat[i].values / sparse_denominator, gradient_flat[i].indices
                    )
                )
            else:
                # Dense gradients.
                accumulator.assign_add(gradient_flat[i] * gradient_flat[i])
                parameter_flat[i].assign_sub(lr * gradient_flat[i] / tf.sqrt(accumulator + self.epsilon))
        return


class Adafactor:
    """Optimizer that implements the Adafactor algorithm.

    Adafactor is commonly used in NLP tasks, and has the advantage
    of taking less memory because it only saves partial information of previous
    gradients.

    The default argument setup is based on the original paper (see reference).
    When gradients are of dimension > 2, Adafactor optimizer will delete the
    last 2 dimensions separately in its accumulator variables.

    Reference:
        - [Shazeer, Noam et al., 2018](https://arxiv.org/abs/1804.04235).

    """

    def __init__(
        self,
        lr=0.001,
        beta_2_decay=-0.8,
        epsilon_1=1e-30,
        epsilon_2=1e-3,
        clip_threshold=1.0,
        relative_step=True,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.beta_2_decay = beta_2_decay
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.clip_threshold = clip_threshold
        self.relative_step = relative_step
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._r = []
        self._c = []
        self._v = []
        self.iterations = tf.Variable(0)
        self.flag = 0


    def _rms(self, x):
        return tf.sqrt(tf.reduce_mean(tf.square(x)))


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        
        if self.flag==0:
            for param in parameter_flat:
                if len(param.shape) < 2:
                    # Don't factor if variable is of dimension < 2, but we still
                    # need to create dummy variables as placeholder.
                    self._r.append(tf.Variable(0, dtype=param.dtype))
                    self._c.append(tf.Variable(0, dtype=param.dtype))
                else:
                    # Always factor the last 2 dimenstions.
                    r_shape = param.shape[:-1]
                    c_shape = param.shape[:-2] + param.shape[-1]
                    self._r.append(
                        tf.Variable(tf.zeros(shape=r_shape,dtype=param.dtype))
                    )
                    self._c.append(
                        tf.Variable(tf.zeros(shape=c_shape,dtype=param.dtype))
                        )
                self._v.append(
                        tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
            self.flag=1
            
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)

        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, parameter_flat[i].dtype)
            epsilon_2 = tf.cast(self.epsilon_2, parameter_flat[i].dtype)
            one = tf.cast(1.0, parameter_flat[i].dtype)
            local_step = tf.cast(self.iterations + 1, parameter_flat[i].dtype)
            if self.relative_step:
                # If `relative_step=True` and learning rate is a constant, we
                # apply the relative step algorithm.
                lr = tf.minimum(lr, tf.math.rsqrt(local_step))
    
            r = self._r[i]
            c = self._c[i]
            v = self._v[i]
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
    
            rho_t = tf.minimum(lr, tf.math.rsqrt(local_step))
            alpha_t = tf.maximum(epsilon_2, self._rms(parameter_flat[i])) * rho_t
            regulated_grad_square = tf.square(gradient_flat[i]) + self.epsilon_1
            beta_2_t = 1 - tf.pow(local_step, self.beta_2_decay)
            
    
            if len(parameter_flat[i].shape) >= 2:
                # `r` deletes the last dimension of gradient, so it is of shape
                # `gradient.shape[:-1]`.
                r.assign(
                    beta_2_t * r
                    + (1 - beta_2_t)
                    * tf.reduce_mean(regulated_grad_square, axis=-1)
                )
                # `c` deletes the second last dimension of gradient, so it is of
                # shape `gradient.shape[:-2] + gradient.shape[-1]`.
                c.assign(
                    beta_2_t * c
                    + (1 - beta_2_t)
                    * tf.reduce_mean(regulated_grad_square, axis=-2)
                )
                v.assign(
                    tf.expand_dims(
                        r / tf.reduce_mean(r, axis=-1, keepdims=True), axis=-1
                    )
                    * tf.expand_dims(c, -2)
                )
            else:
                v.assign(beta_2_t * v + (1 - beta_2_t) * regulated_grad_square)
    
            # `convert_to_tensor` unifies the handling of sparse and dense grads.
            u_t = tf.convert_to_tensor(gradient_flat[i]) * tf.math.rsqrt(v)
            u_t_hat = u_t / tf.maximum(one, (self._rms(u_t) / self.clip_threshold))
            parameter_flat[i].assign_add(-alpha_t * u_t_hat)
        self.iterations.assign_add(1)
        return

    

class RMSprop:
    r"""Optimizer that implements the RMSprop algorithm.

    The gist of RMSprop is to:

    - Maintain a moving (discounted) average of the square of gradients
    - Divide the gradient by the root of this average

    This implementation of RMSprop uses plain momentum, not Nesterov momentum.

    The centered version additionally maintains a moving average of the
    gradients, and uses that average to estimate the variance.

    Reference:
        - [Hinton, 2012](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) # noqa: E501
    """

    def __init__(
        self,
        lr=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        centered=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._velocities = []
        self._momentums = []
        self._average_gradients = []
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        
        if self.flag==0:
            for param in parameter_flat:
                self._velocities.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
    
            if self.momentum > 0:
                for param in parameter_flat:
                    self._momentums.append(
                        tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                    )
    
            if self.centered:
                for param in parameter_flat:
                    self._average_gradients.append(
                        tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                    )
            self.flag=1
            
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)

        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, dtype=parameter_flat[i].dtype)
            
            velocity = self._velocities[i]
            momentum = None
            if self.momentum > 0:
                momentum = self._momentums[i]
            average_grad = None
            if self.centered:
                average_grad = self._average_gradients[i]
    
            rho = self.rho
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
            
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                velocity.assign(rho * velocity)
                velocity.scatter_add(
                    tf.IndexedSlices(
                        tf.square(gradient_flat[i].values) * (1 - rho), gradient_flat[i].indices
                    )
                )
                if self.centered:
                    average_grad.assign(rho * average_grad)
                    average_grad.scatter_add(
                        tf.IndexedSlices(
                            gradient_flat[i].values * (1 - rho), gradient_flat[i].indices
                        )
                    )
                    denominator = velocity - tf.square(average_grad) + self.epsilon
                else:
                    denominator = velocity + self.epsilon
                denominator_slices = tf.gather(denominator, gradient_flat[i].indices)
                increment = tf.IndexedSlices(
                    lr * gradient_flat[i].values * tf.math.rsqrt(denominator_slices),
                    gradient_flat[i].indices,
                )
    
                if self.momentum > 0:
                    momentum.assign(self.momentum * momentum)
                    momentum.scatter_add(increment)
                    parameter_flat[i].assign_add(-momentum)
                else:
                    parameter_flat[i].scatter_add(-increment)
            else:
                # Dense gradients.
                velocity.assign(rho * velocity + (1 - rho) * tf.square(gradient_flat[i]))
                if self.centered:
                    average_grad.assign(rho * average_grad + (1 - rho) * gradient_flat[i])
                    denominator = velocity - tf.square(average_grad) + self.epsilon
                else:
                    denominator = velocity + self.epsilon
                increment = lr * gradient_flat[i] * tf.math.rsqrt(denominator)
                if self.momentum > 0:
                    momentum.assign(self.momentum * momentum + increment)
                    parameter_flat[i].assign_add(-momentum)
                else:
                    parameter_flat[i].assign_add(-increment)
        return


class Adadelta:
    r"""Optimizer that implements the Adadelta algorithm.

    Adadelta optimization is a stochastic gradient descent method that is based
    on adaptive learning rate per dimension to address two drawbacks:

    - The continual decay of learning rates throughout training.
    - The need for a manually selected global learning rate.

    Adadelta is a more robust extension of Adagrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past gradients. This way, Adadelta continues learning even when many updates
    have been done. Compared to Adagrad, in the original version of Adadelta you
    don't have to set an initial learning rate. In this version, the initial
    learning rate can be set, as in most other Keras optimizers.

    Reference:
        - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
    """

    def __init__(
        self,
        lr=0.001,
        rho=0.95,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._accumulated_grads = []
        self._accumulated_delta_vars = []
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        
        if self.flag==0:
            for param in parameter_flat:
                self.accumulated_grads.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
                self.accumulated_delta_vars.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
            self.flag=1
            
        rho = self.rho

        def rms(x):
            return tf.sqrt(x + self.epsilon)
        
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)

        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, dtype=parameter_flat[i].dtype)
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
            
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                self.accumulated_grad[i].assign_add((rho - 1) * self.accumulated_grad[i])
                self.accumulated_grad[i].scatter_add(
                    tf.IndexedSlices(
                        (1 - rho) * tf.square(gradient_flat[i].values), gradient_flat[i].indices
                    )
                )
                delta_var = (
                    -rms(self.accumulated_delta_var[i]) * gradient_flat[i] / rms(self.accumulated_grad[i])
                )
                self.accumulated_delta_var[i].assign(
                    rho * self.accumulated_delta_var[i] + (1 - rho) * delta_var * delta_var
                )
            else:
                # Dense gradients.
                self.accumulated_grad[i].assign(
                    rho * self.accumulated_grad[i] + (1 - rho) * gradient_flat[i] * gradient_flat[i]
                )
                delta_var = (
                    -rms(self.accumulated_delta_var[i]) * gradient_flat[i] / rms(self.accumulated_grad[i])
                )
                self.accumulated_delta_var[i].assign(
                    rho * self.accumulated_delta_var[i] + (1 - rho) * delta_var * delta_var
                )
            parameter_flat[i].assign_add(lr * delta_var)
        return


class Adam:
    r"""Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Reference:
        - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
        - [Reddi et al., 2018](
            https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since Adam uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._momentums = []
        self._velocities = []
        if self.amsgrad:
            self._velocity_hats = []
        self.iterations = tf.Variable(0)
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            for param in parameter_flat:
                self._momentums.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
                self._velocities.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
                if self.amsgrad:
                    for param in parameter_flat:
                        self._velocity_hats.append(
                            tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                        )
            self.flag=1
            
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
                            
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, dtype=parameter_flat[i].dtype)
            
            local_step = tf.cast(self.iterations + 1, parameter_flat[i].dtype)
            beta_1_power = tf.pow(tf.cast(self.beta_1, parameter_flat[i].dtype), local_step)
            beta_2_power = tf.pow(tf.cast(self.beta_2, parameter_flat[i].dtype), local_step)
    
            m = self._momentums[i]
            v = self._velocities[i]
    
            alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
    
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                m.assign_add(-m * (1 - self.beta_1))
                m.scatter_add(
                    tf.IndexedSlices(
                        gradient_flat[i].values * (1 - self.beta_1), gradient_flat[i].indices
                    )
                )
                v.assign_add(-v * (1 - self.beta_2))
                v.scatter_add(
                    tf.IndexedSlices(
                        tf.square(gradient_flat[i].values) * (1 - self.beta_2),
                        gradient_flat[i].indices,
                    )
                )
                if self.amsgrad:
                    v_hat = self._velocity_hats[i]
                    v_hat.assign(tf.maximum(v_hat, v))
                    v = v_hat
                parameter_flat[i].assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
            else:
                # Dense gradients.
                m.assign_add((gradient_flat[i] - m) * (1 - self.beta_1))
                v.assign_add((tf.square(gradient_flat[i]) - v) * (1 - self.beta_2))
                if self.amsgrad:
                    v_hat = self._velocity_hats[i]
                    v_hat.assign(tf.maximum(v_hat, v))
                    v = v_hat
                parameter_flat[i].assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        self.iterations.assign_add(1)
        return


class Nadam:
    r"""Optimizer that implements the Nadam algorithm.

    Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
    Nesterov momentum.

    Reference:
        - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).

    """

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._momentums = []
        self._velocities = []
        self._u_product = []
        # Keep a counter on how many times of _u_product has been computed to
        # avoid duplicated computations.
        self._u_product_counter = 1
        self.iterations = tf.Variable(0)
        self.flag = 0
                

    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            for param in parameter_flat:
                self._u_product.append(tf.Variable(1.0, param.dtype))
                self._momentums.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
                self._velocities.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
            self.flag=1
        
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
        
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            var_dtype = parameter_flat[i].dtype
            lr = tf.cast(self.lr, var_dtype)
            local_step = tf.cast(self.iterations + 1, var_dtype)
            next_step = tf.cast(self.iterations + 2, var_dtype)
            decay = tf.cast(0.96, var_dtype)
            beta_1 = tf.cast(self.beta_1, var_dtype)
            beta_2 = tf.cast(self.beta_2, var_dtype)
            u_t = beta_1 * (1.0 - 0.5 * (tf.pow(decay, local_step)))
            u_t_1 = beta_1 * (1.0 - 0.5 * (tf.pow(decay, next_step)))
    
            if self._u_product_counter == (self.iterations + 2):
                 u_product_t = self._u_product[i]
            else:
                u_product_t = self._u_product[i] * u_t
                self._u_product.assign(u_product_t)
                self._u_product_counter += 1
                
            u_product_t_1 = u_product_t * u_t_1
            beta_2_power = tf.pow(beta_2, local_step)
    
            m = self._momentums[i]
            v = self._velocities[i]
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
            
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                m.assign_add(-m * (1 - beta_1))
                m.scatter_add(
                    tf.IndexedSlices(
                        gradient_flat[i].values * (1 - beta_1), gradient_flat[i].indices
                    )
                )
                v.assign_add(-v * (1 - beta_2))
                v.scatter_add(
                    tf.IndexedSlices(
                        tf.square(gradient_flat[i].values) * (1 - beta_2), gradient_flat[i].indices
                    )
                )
                m_hat = u_t_1 * m / (1 - u_product_t_1) + (1 - u_t) * gradient_flat[i] / (
                    1 - u_product_t
                )
                v_hat = v / (1 - beta_2_power)
    
                parameter_flat[i].assign_sub((m_hat * lr) / (tf.sqrt(v_hat) + self.epsilon))
            else:
                # Dense gradients.
                m.assign_add((gradient_flat[i] - m) * (1 - beta_1))
                v.assign_add((tf.square(gradient_flat[i]) - v) * (1 - beta_2))
                m_hat = u_t_1 * m / (1 - u_product_t_1) + (1 - u_t) * gradient_flat[i] / (
                    1 - u_product_t
                )
                v_hat = v / (1 - beta_2_power)
    
                parameter_flat[i].assign_sub((m_hat * lr) / (tf.sqrt(v_hat) + self.epsilon))
        self.iterations.assign_add(1)
        return


class Adamax:
    """Optimizer that implements the Adamax algorithm.

    Adamax, a variant of Adam based on the infinity norm, is a first-order
    gradient-based optimization method. Due to its capability of adjusting the
    learning rate based on data characteristics, it is suited to learn
    time-variant process, e.g., speech data with dynamically changed noise
    conditions. Default parameters follow those provided in the paper (see
    references below).

    Initialization:

    ```python
    m = 0  # Initialize initial 1st moment vector
    u = 0  # Initialize the exponentially weighted infinity norm
    t = 0  # Initialize timestep
    ```

    The update rule for parameter `w` with gradient `g` is described at the end
    of section 7.1 of the paper (see the referenece section):

    ```python
    t += 1
    m = beta1 * m + (1 - beta) * g
    u = max(beta2 * u, abs(g))
    current_lr = learning_rate / (1 - beta1 ** t)
    w = w - current_lr * m / (u + epsilon)
    ```
    
    Reference:
        - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    """

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._m = []
        self._u = []
        self.iterations = tf.Variable(0)
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            for param in parameter_flat:
                self._m.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
                self._u.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
            self.flag=1
            
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
        
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, dtype=parameter_flat[i].dtype)
            
            local_step = tf.cast(self.iterations + 1, parameter_flat[i].dtype)
            beta_1_power = tf.pow(tf.cast(self.beta_1, parameter_flat[i].dtype), local_step)
    
            m = self._m[i]
            u = self._u[i]
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
            
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                indices = gradient_flat[i].indices
                m.assign_add(-m * (1 - self.beta_1))
                m.scatter_add(
                    tf.IndexedSlices(gradient_flat[i].values * (1 - self.beta_1), indices)
                )
                u.assign(u * self.beta_2)
                u_slice = tf.gather(u, indices)
                u_slice_incremental = (
                    tf.maximum(u_slice, tf.abs(gradient_flat[i].values)) - u_slice
                )
                u.scatter_add(tf.IndexedSlices(u_slice_incremental, indices))
                parameter_flat[i].assign_sub(
                    (lr * m) / ((1 - beta_1_power) * (u + self.epsilon))
                )
            else:
                # Dense gradients.
                m.assign_add((gradient_flat[i] - m) * (1 - self.beta_1))
                u.assign(tf.maximum(self.beta_2 * u, tf.abs(gradient_flat[i])))
                parameter_flat[i].assign_sub(
                    (lr * m) / ((1 - beta_1_power) * (u + self.epsilon))
                )
        self.iterations.assign_add(1)
        return


class AdamW:
    r"""Optimizer that implements the AdamW algorithm.

    AdamW optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments with an added
    method to decay weights per the techniques discussed in the paper,
    'Decoupled Weight Decay Regularization' by
    [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the underying Adam method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Reference:
      - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980) for `adam`
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        lr=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._momentums = []
        self._velocities = []
        if self.amsgrad:
            self._velocity_hats = []
        self.iterations = tf.Variable(0)
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            for param in parameter_flat:
                self._momentums.append(
                        tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
                self._velocities.append(
                        tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
            if self.amsgrad:
                for param in parameter_flat:
                    self._velocity_hats.append(
                        tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                    )
            self.flag=1
            
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
                    
        for i in range(len(gradient_flat)):  
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, dtype=parameter_flat[i].dtype)
            
            local_step = tf.cast(self.iterations + 1, parameter_flat[i].dtype)
            beta_1_power = tf.pow(tf.cast(self.beta_1, parameter_flat[i].dtype), local_step)
            beta_2_power = tf.pow(tf.cast(self.beta_2, parameter_flat[i].dtype), local_step)
    
            m = self._momentums[i]
            v = self._velocities[i]
    
            alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
    
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients.
                m.assign_add(-m * (1 - self.beta_1))
                m.scatter_add(
                    tf.IndexedSlices(
                        gradient_flat[i].values * (1 - self.beta_1), gradient_flat[i].indices
                    )
                )
                v.assign_add(-v * (1 - self.beta_2))
                v.scatter_add(
                    tf.IndexedSlices(
                        tf.square(gradient_flat[i].values) * (1 - self.beta_2),
                        gradient_flat[i].indices,
                    )
                )
                if self.amsgrad:
                    v_hat = self._velocity_hats[i]
                    v_hat.assign(tf.maximum(v_hat, v))
                    v = v_hat
                parameter_flat[i].assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
            else:
                # Dense gradients.
                m.assign_add((gradient_flat[i] - m) * (1 - self.beta_1))
                v.assign_add((tf.square(gradient_flat[i]) - v) * (1 - self.beta_2))
                if self.amsgrad:
                    v_hat = self._velocity_hats[i]
                    v_hat.assign(tf.maximum(v_hat, v))
                    v = v_hat
                parameter_flat[i].assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        self.iterations.assign_add(1)
        return


class Ftrl:
    r"""Optimizer that implements the FTRL algorithm.

    "Follow The Regularized Leader" (FTRL) is an optimization algorithm
    developed at Google for click-through rate prediction in the early 2010s. It
    is most suitable for shallow models with large and sparse feature spaces.
    The algorithm is described by
    [McMahan et al., 2013](https://research.google.com/pubs/archive/41159.pdf).
    The Keras version has support for both online L2 regularization
    (the L2 regularization described in the paper
    above) and shrinkage-type L2 regularization
    (which is the addition of an L2 penalty to the loss function).

    Initialization:

    ```python
    n = 0
    sigma = 0
    z = 0
    ```

    Update rule for one variable `w`:

    ```python
    prev_n = n
    n = n + g ** 2
    sigma = (n ** -lr_power - prev_n ** -lr_power) / lr
    z = z + g - sigma * w
    if abs(z) < lambda_1:
      w = 0
    else:
      w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
    ```

    Notation:

    - `lr` is the learning rate
    - `g` is the gradient for the variable
    - `lambda_1` is the L1 regularization strength
    - `lambda_2` is the L2 regularization strength
    - `lr_power` is the power to scale n.
    """

    def __init__(
        self,
        lr=0.001,
        learning_rate_power=-0.5,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0,
        l2_shrinkage_regularization_strength=0.0,
        beta=0.0,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        if initial_accumulator_value < 0.0:
            raise ValueError(
                "`initial_accumulator_value` needs to be positive or zero. "
                "Received: initial_accumulator_value="
                f"{initial_accumulator_value}."
            )
        if learning_rate_power > 0.0:
            raise ValueError(
                "`learning_rate_power` needs to be negative or zero. Received: "
                f"learning_rate_power={learning_rate_power}."
            )
        if l1_regularization_strength < 0.0:
            raise ValueError(
                "`l1_regularization_strength` needs to be positive or zero. "
                "Received: l1_regularization_strength="
                f"{l1_regularization_strength}."
            )
        if l2_regularization_strength < 0.0:
            raise ValueError(
                "`l2_regularization_strength` needs to be positive or zero. "
                "Received: l2_regularization_strength="
                f"{l2_regularization_strength}."
            )
        if l2_shrinkage_regularization_strength < 0.0:
            raise ValueError(
                "`l2_shrinkage_regularization_strength` needs to be positive "
                "or zero. Received: l2_shrinkage_regularization_strength"
                f"={l2_shrinkage_regularization_strength}."
            )
        self.lr = lr
        self.learning_rate_power = learning_rate_power
        self.initial_accumulator_value = initial_accumulator_value
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.l2_shrinkage_regularization_strength = (
            l2_shrinkage_regularization_strength
        )
        self.beta = beta
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self._accumulators = []
        self._linears = []
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            for param in parameter_flat:
                self._accumulators.append(tf.Vriable(tf.fill(dims=param.shape, 
                                                    value=self.initial_accumulator_value),
                                                     dtype=param.dtype))
                self._linears.append(tf.Variable(tf.zeros_like(param,dtype=param.dtype)))
            self.flag=1
            
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
        
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.lr, parameter_flat[i].dtype)
            accum = self._accumulators[i]
            linear = self._linears[i]
    
            lr_power = self.learning_rate_power
            l2_reg = self.l2_regularization_strength
            l2_reg = l2_reg + self.beta / (2.0 * lr)
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
    
            # Ftrl optimizer has the same implementation for sparse and dense
            # gradients update.
            grad_to_use = (
                gradient_flat[i] + 2 * self.l2_shrinkage_regularization_strength * parameter_flat[i]
            )
            new_accum = accum + tf.pow(gradient_flat[i], 2)
            linear.assign_add(
                grad_to_use
                - (tf.pow(new_accum, -lr_power) - tf.pow(accum, -lr_power))
                / lr
                * parameter_flat[i]
            )
            quadratic = tf.pow(new_accum, (-lr_power)) / lr + 2 * l2_reg
            linear_clipped = tf.clip_by_value(
                linear,
                -self.l1_regularization_strength,
                self.l1_regularization_strength,
            )
            parameter_flat[i].assign((linear_clipped - linear) / quadratic)
            accum.assign(new_accum)
        return


class Lion:
    """Optimizer that implements the Lion algorithm.

    The Lion optimizer is a stochastic-gradient-descent method that uses the
    sign operator to control the magnitude of the update, unlike other adaptive
    optimizers such as Adam that rely on second-order moments. This make
    Lion more memory-efficient as it only keeps track of the momentum. According
    to the authors (see reference), its performance gain over Adam grows with
    the batch size. Because the update of Lion is produced through the sign
    operation, resulting in a larger norm, a suitable learning rate for Lion is
    typically 3-10x smaller than that for AdamW. The weight decay for Lion
    should be in turn 3-10x larger than that for AdamW to maintain a
    similar strength (lr * wd).

    References:
        - [Chen et al., 2023](http://arxiv.org/abs/2302.06675)
        - [Authors' implementation](
            http://github.com/google/automl/tree/master/lion)

    """

    def __init__(
        self,
        lr=0.0001,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        if beta_1 <= 0 or beta_1 > 1:
            raise ValueError(
                f"`beta_1`={beta_1} must be between ]0, 1]. Otherwise, "
                "the optimizer degenerates to SignSGD."
            )
        self.momentums = []
        self.flag = 0


    def __call__(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            for param in parameter_flat:
                self.momentums.append(
                        tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                        )
            self.flag=1
            
        if self.weight_decay!=None and self.global_clipnorm!=None:
            wd_p = [tf.cast(self.weight_decay, p.dtype) * p for p in parameter_flat]
            gradient_flat = [g + wd_p[i] for i,g in enumerate(gradient_flat)]
            gradient_flat, _ = tf.clip_by_global_norm(gradient_flat, self.global_clipnorm)
                
        for i in range(len(gradient_flat)):
            if parameter_flat[i].trainable!=True:
                continue
            
            lr = tf.cast(self.learning_rate, parameter_flat[i].dtype)
            beta_1 = tf.cast(self.beta_1, parameter_flat[i].dtype)
            beta_2 = tf.cast(self.beta_2, parameter_flat[i].dtype)
            m = self.momentums[i]
            
            if self.weight_decay!=None and self.global_clipnorm==None:
                wd = tf.cast(self.weight_decay, parameter_flat[i].dtype)
                gradient_flat[i] = gradient_flat[i] + wd * parameter_flat[i]
            if self.clipnorm!=None:
                gradient_flat[i] = tf.clip_by_norm(gradient_flat[i], self.clipnorm)
            if self.clipvalue!=None:
                gradient_flat[i] = tf.clip_by_value(gradient_flat[i], -self.clipvalue, self.clipvalue)
    
            if isinstance(gradient_flat[i], tf.IndexedSlices):
                # Sparse gradients (use m as a buffer)
                m.assign(m * beta_1)
                m.scatter_add(
                    tf.IndexedSlices(
                        gradient_flat[i].values * (1.0 - beta_1), gradient_flat[i].indices
                    )
                )
                parameter_flat[i].assign_sub(lr * tf.math.sign(m))
    
                m.assign(m * beta_2 / beta_1)
                m.scatter_add(
                    tf.IndexedSlices(
                        gradient_flat[i].values * (1.0 - beta_2 / beta_1), gradient_flat[i].indices
                    )
                )
            else:
                # Dense gradients
                parameter_flat[i].assign_sub(
                    lr * tf.math.sign(m * beta_1 + gradient_flat[i] * (1.0 - beta_1))
                )
                m.assign(m * beta_2 + gradient_flat[i] * (1.0 - beta_2))
        return
