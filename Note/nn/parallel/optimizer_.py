import tensorflow as tf
from tensorflow.python.util import nest


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
        param=None
    ):
        self.lr = tf.Variable(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")
        self.momentums = []
        self.flag=0


    def opt(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.flag=1
            parameter_flat=nest.flatten(parameter)
            for param in parameter_flat:
                self.momentums.append(
                    tf.Variable(tf.zeros_like(param,dtype=param.dtype))
                )
        
        for i in range(len(gradient_flat)):
            lr = tf.cast(self.lr, parameter_flat[i].dtype)
            m = None
            momentum = tf.cast(self.momentum, parameter_flat[i].dtype)
            m = self.momentums[i]
    
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
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter


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
        param=None
    ):
        self.lr = tf.Variable(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self._momentums = []
        self._velocities = []
        if self.amsgrad:
            self._velocity_hats = []
        self.iterations=tf.Variable(0)
        self.flag=0


    def opt(self, gradient, parameter):
        gradient_flat=nest.flatten(gradient)
        parameter_flat=nest.flatten(parameter)
        if self.flag==0:
            self.flag=1
            parameter_flat=nest.flatten(parameter)
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
                            
        for i in range(len(gradient_flat)):
            iterations=self.iterations
            lr = tf.cast(self.lr, dtype=parameter_flat[i].dtype)
            
            local_step = tf.cast(iterations + 1, parameter_flat[i].dtype)
            beta_1_power = tf.pow(tf.cast(self.beta_1, parameter_flat[i].dtype), local_step)
            beta_2_power = tf.pow(tf.cast(self.beta_2, parameter_flat[i].dtype), local_step)
    
            m = self._momentums[i]
            v = self._velocities[i]
    
            alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
    
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
        parameter=nest.pack_sequence_as(parameter,parameter_flat)
        return parameter