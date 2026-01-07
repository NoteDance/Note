import re
import warnings

from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.saving.keras_saveable import KerasSaveable
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name

import tensorflow as tf
import math


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


def unit_norm(x, ord = 2.0):
    r"""Get norm of unit."""
    keepdims = True
    axis = None

    x_len = len(x.shape)
    if x_len <= 1:
        keepdims = False
    elif x_len in (2, 3):
        axis = 1
    elif x_len == 4:
        axis = (1, 2, 3)
    else:
        axis = tuple(range(1, x_len))

    return tf.norm(x, ord=ord, axis=axis, keepdims=keepdims)


class BaseOptimizer(KerasSaveable):
    """Abstract optimizer base class.

    If you intend to create your own optimization algorithm, please inherit from
    this class and override the following methods:

    - `build`: Create your optimizer-related variables, such as momentum
        variables in the SGD optimizer.
    - `update_step`: Implement your optimizer's variable updating logic.
    - `get_config`: serialization of the optimizer.

    Example:

    ```python
    class SGD(Optimizer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.momentum = 0.9

        def build(self, variables):
            super().build(variables)
            self.momentums = []
            for variable in variables:
                self.momentums.append(
                    self.add_variable_from_reference(
                        reference_variable=variable, name="momentum"
                    )
                )

        def update_step(self, gradient, variable, learning_rate):
            learning_rate = ops.cast(learning_rate, variable.dtype)
            gradient = ops.cast(gradient, variable.dtype)
            m = self.momentums[self._get_variable_index(variable)]
            self.assign(
                m,
                ops.subtract(
                    ops.multiply(m, ops.cast(self.momentum, variable.dtype)),
                    ops.multiply(gradient, learning_rate),
                ),
            )
            self.assign_add(variable, m)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                }
            )
            return config
    ```
    """

    def __init__(
        self,
        learning_rate,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs,
    ):
        self._lock = False

        if kwargs.pop("decay", None) is not None:
            warnings.warn(
                "Argument `decay` is no longer supported and will be ignored."
            )
        if kwargs:
            raise ValueError(f"Argument(s) not recognized: {kwargs}")

        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.clipvalue = clipvalue
        self.use_ema = use_ema
        self.loss_scale_factor = loss_scale_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if gradient_accumulation_steps:
            if not gradient_accumulation_steps >= 2:
                raise ValueError(
                    "`gradient_accumulation_steps` must be an integer >= 2. "
                    "Received: gradient_accumulation_steps="
                    f"{gradient_accumulation_steps}"
                )

        if use_ema:
            # Verify the arguments related to EMA.
            if ema_momentum > 1 or ema_momentum < 0:
                raise ValueError(
                    "`ema_momentum` must be in the range [0, 1]. "
                    f"Received: ema_momentum={ema_momentum}"
                )
            if ema_overwrite_frequency and (
                not isinstance(ema_overwrite_frequency, int)
                or ema_overwrite_frequency < 1
            ):
                raise ValueError(
                    "`ema_overwrite_frequency` must be an integer >= 1 or "
                    "None. Received: ema_overwrite_frequency="
                    f"{ema_overwrite_frequency}"
                )
        self.ema_momentum = ema_momentum
        self.ema_overwrite_frequency = ema_overwrite_frequency

        clip_args_sum = sum(
            a is not None for a in [clipnorm, clipvalue, global_clipnorm]
        )
        if clip_args_sum > 1:
            raise ValueError(
                "Only one of `clipnorm`, `clipvalue` and `global_clipnorm` can "
                f"be set. Received: clipnorm={clipnorm}, "
                f"clipvalue={clipvalue}, global_clipnorm={global_clipnorm}"
            )
        self.built = False

        # Set up variable tracking.
        self._variables = []
        self._trainable_variables = []
        self._tracker = tracking.Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    self._variables,
                ),
            }
        )
        self._trainable_variables_indices = {}

        # Create iteration variable
        # Note: dtype="int" will resolve to int32 in JAX
        # (since int64 is disallowed in JAX) and to int64 in TF.
        with backend.name_scope(self.name, caller=self):
            iterations = backend.Variable(
                0,
                name="iteration",
                dtype="int",
                trainable=False,
                aggregation="only_first_replica",
            )
        self._track_variable(iterations)
        self._iterations = iterations

        # Create learning rate (schedule or variable)
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if not isinstance(learning_rate, float):
                raise ValueError(
                    "Argument `learning_rate` should be float, or an instance "
                    "of LearningRateSchedule, or a callable "
                    "(that takes in the current iteration value "
                    "and returns the corresponding learning rate value). "
                    f"Received instead: learning_rate={learning_rate}"
                )
            with backend.name_scope(self.name, caller=self):
                learning_rate = backend.Variable(
                    learning_rate,
                    name="learning_rate",
                    dtype=backend.floatx(),
                    trainable=False,
                    aggregation="only_first_replica",
                )
            self._track_variable(learning_rate)
            self._learning_rate = learning_rate

    @property
    def iterations(self):
        if self.gradient_accumulation_steps:
            return ops.floor_divide(
                self._iterations, self.gradient_accumulation_steps
            )

        return self._iterations

    def _track_variable(self, variable):
        self._tracker.add_to_store("variables", variable)

    def _overwrite_variable_with_gradient(self, variable):
        return getattr(variable, "overwrite_with_gradient", False)

    @tracking.no_automatic_dependency_tracking
    def build(self, variables):
        if self.sn:
            self.subset_size_ = []
        if self.sophia:
            self.hessian_moment = []
            self.hessian = []
        if self.lookahead:
            self.slow_momentum = []
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
        if self.pnm:
            self.pos_momentum = []
            self.neg_momentum = []
        if self.use_ema:
            self._model_variables_moving_average = self.add_optimizer_variables(
                variables, "average"
            )
        if self.gradient_accumulation_steps:
            self._accumulated_gradients = []
        for i, variable in enumerate(variables):
            self._trainable_variables_indices[self._var_key(variable)] = i
            if self.gradient_accumulation_steps:
                self._accumulated_gradients.append(
                    self.add_variable_from_reference(
                        variable,
                        name="gradient_accumulator",
                    )
                )
            if self.sn:
                size = tf.size(variable)
                
                def true_fn():
                    return self.subset_size
                def false_fn():
                    return tf.cast(tf.sqrt(size) / tf.abs(tf.cast(self.subset_size, tf.int32)), tf.int32)
                self.subset_size_.append(closest_smaller_divisor_of_n_to_k(
                    size,
                    tf.cond(self.subset_size > 0, true_fn, false_fn)
                ))

                reshaped_grad = tf.reshape(variable, (size // self.subset_size_[-1], self.subset_size_[-1]))
                second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)  # fmt: skip
                second_moment_update = tf.Variable(second_moment_update)
                if self.sophia:
                    self.hessian[self._get_variable_index(variable)] =  self.add_variable_from_reference(
                                                                reference_variable=second_moment_update, name="hessian"
                                                            )
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=second_moment_update, name="hessian_moment"
                                            ))
                else:
                    self.exp_avg_sq.append(self.add_variable_from_reference(
                            reference_variable=second_moment_update, name="exp_avg_sq"
                        ))
            else:
                if self.sophia:
                    self.hessian[self._get_variable_index(variable)] =  self.add_variable_from_reference(
                                                                reference_variable=variable, name="hessian"
                                                            )
                    self.hessian_moment.append(self.add_variable_from_reference(
                        reference_variable=variable, name="hessian_moment"
                                            ))
                else:
                    self.exp_avg_sq.append(self.add_variable_from_reference(
                        reference_variable=variable, name="exp_avg_sq"
                    ))
            
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(variable))
                self._track_variable(self.slow_momentum[-1])
                
            if self.DAdapt:
                self.s.append(self.add_variable_from_reference(
                                    reference_variable=variable, name="s"
                                                        ))
            
            if self.pnm:
                self.pos_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=variable, name="pos_momentum"
                    )
                )
                self.neg_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=variable, name="neg_momentum"
                    )
                )
        self._trainable_variables = variables[:]
        self.built = True

    def _var_key(self, variable):
        # Helper function to get a stable ID and the variable instance mapping.
        return id(variable)

    @property
    def variables(self):
        return self._variables[:]

    def _get_variable_index(self, variable):
        return self._trainable_variables_indices[self._var_key(variable)]

    def add_variable(
        self,
        shape,
        initializer="zeros",
        dtype=None,
        aggregation="none",
        layout=None,
        name=None,
    ):
        """Add a variable to the optimizer.

        Args:
            shape: Shape tuple for the variable. Must be fully-defined
                (no `None` entries).
            initializer: Initializer object to use to populate the initial
                variable value, or string name of a built-in initializer
                (e.g. `"random_normal"`). Defaults to `"zeros"`.
            dtype: Dtype of the variable to create, e.g. `"float32"`. If
                unspecified, defaults to the `keras.backend.floatx()`.
            aggregation: Optional string, one of `None`, `"none"`, `"mean"`,
                `"sum"` or `"only_first_replica"`. Annotates the variable with
                the type of multi-replica aggregation to be used for this
                variable when writing custom data parallel training loops.
                Defaults to `"none"`.
            layout: Optional tensor layout.  Defaults to `None`.
            name: String name of the variable. Useful for debugging purposes.

        Returns:
            An optimizer variable, in the format of `keras.Variable`.
        """
        self._check_super_called()
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=False,
                aggregation=aggregation,
                layout=layout,
                name=name,
            )
        self._track_variable(variable)
        return variable

    def add_variable_from_reference(
        self, reference_variable, name=None, initializer="zeros"
    ):
        """Add an optimizer variable from the model variable.

        Create an optimizer variable based on the information of model variable.
        For example, in SGD optimizer momemtum, for each model variable, a
        corresponding momemtum variable is created of the same shape and dtype.

        Args:
            reference_variable: `keras.Variable`. The corresponding model
                variable to the optimizer variable to be created.
            name: Optional string. The name prefix of the optimizer variable to
                be created. If not provided, it will be set to `"var"`. The
                variable name will follow the pattern
                `{variable_name}_{reference_variable.name}`,
                e.g., `momemtum/dense_1`. Defaults to `None`.
            initializer: Initializer object to use to populate the initial
                variable value, or string name of a built-in initializer
                (e.g. `"random_normal"`). If unspecified, defaults to
                `"zeros"`.

        Returns:
            An optimizer variable, in the format of `keras.Variable`.
        """
        name = name or "var"
        if hasattr(reference_variable, "path"):
            name = f"{reference_variable.path.replace('/', '_')}_{name}"
        else:
            sanitised_ref_name = (
                str(reference_variable.name).replace("/", "_").replace(":", "_")
            )
            name = f"{sanitised_ref_name}_{name}"
        return self.add_variable(
            shape=reference_variable.shape,
            initializer=initializer,
            dtype=reference_variable.dtype,
            name=name,
            layout=getattr(reference_variable, "_layout", None),
        )

    def add_optimizer_variables(
        self, trainable_variables, name, initializer="zeros"
    ):
        """Add optimizer variables from the list of trainable model variables.

        Create an optimizer variable based on the information of the supplied
        model variables.  For example, in SGD optimizer momemtum, for each model
        variable, a corresponding momemtum variable is created of the same shape
        and dtype.

        Note that trainable variables with `v.overwrite_with_gradient == True`
        will insert `None`, into the output list, since the optimizer variable
        will not be used anyways, and could be wasteful.

        Args:
            trainable_variables: `keras.Variable`, the corresponding model
                variable to the optimizer variable to be created.
            name: The name prefix(es) of the optimizer variable(s) to be
                created. Can be a single string or list of strings.  If a
                list of strings, will create an optimizer variable for each
                prefix.  The variable name will follow the pattern
                `{variable_name}_{trainable_variable.name}`, e.g.,
                `momemtum/dense_1`.
            initializer: Initializer object(s) to use to populate the initial
                variable value(s), or string name of a built-in initializer
                (e.g. `"random_normal"`). If unspecified, defaults to
                `"zeros"`.

        Returns:
            A list of optimizer variables, in the format of `keras.Variable`s.
            If multiple names are provide, returns a tuple of lists.
        """
        name_list = name
        initializer_list = initializer
        if isinstance(name, str):
            # Single name/initializer.
            name_list = [name]
            initializer_list = [initializer]
        else:
            # Multiple names/initializers.
            # If there is only one initializer, use it for all names.
            if isinstance(initializer, str) or isinstance(
                initializer, initializers.Initializer
            ):
                initializer_list = [initializer] * len(name_list)

        if len(name_list) != len(initializer_list):
            raise ValueError(
                f"The number of provided names must match the number of "
                f"provided initializers.  Received name='{name}', "
                f"initializer='{initializer}'"
            )

        # Build up lists of optimizer variables.
        optimizer_variables = tuple([] for _ in name_list)
        for variable in trainable_variables:
            # Interleaves adding variables for backward-compatibility.
            if not self._overwrite_variable_with_gradient(variable):
                for i, (var_name, var_init) in enumerate(
                    zip(name_list, initializer_list)
                ):
                    optimizer_variables[i].append(
                        self.add_variable_from_reference(
                            variable,
                            name=var_name,
                            initializer=var_init,
                        )
                    )
            else:
                for i in range(len(name_list)):
                    optimizer_variables[i].append(None)

        # If single input name, return the single list.
        if isinstance(name, str):
            return optimizer_variables[0]

        return optimizer_variables

    def _check_variables_are_known(self, variables):
        for v in variables:
            if self._var_key(v) not in self._trainable_variables_indices:
                raise ValueError(
                    f"Unknown variable: {v}. This optimizer can only "
                    "be called for the variables it was originally built with. "
                    "When working with a new set of variables, you should "
                    "recreate a new optimizer instance."
                )

    def assign(self, variable, value):
        """Assign a value to a variable.

        This should be used in optimizers instead of `variable.assign(value)` to
        support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign(value)

    def assign_add(self, variable, value):
        """Add a value to a variable.

        This should be used in optimizers instead of
        `variable.assign_add(value)` to support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign_add(value)

    def assign_sub(self, variable, value):
        """Subtract a value from a variable.

        This should be used in optimizers instead of
        `variable.assign_sub(value)` to support backend specific optimizations.
        Note that the variable can be a model variable or an optimizer variable;
        it can be a backend native variable or a Keras variable.

        Args:
            variable: The variable to update.
            value: The value to add to the variable.
        """
        variable.assign_sub(value)
    
    def agc(
        self, p, grad, agc_eps = 1e-3, agc_clip_val = 1e-2, eps = 1e-6
    ):
        r"""Clip gradient values in excess of the unit wise norm."""
        max_norm = tf.maximum(unit_norm(p), agc_eps) * agc_clip_val
        g_norm = tf.maximum(unit_norm(grad), eps)
    
        clipped_grad = grad * (max_norm / g_norm)
    
        return tf.where(g_norm > max_norm, clipped_grad, grad)
    
    def gc(self, grads, gradient, idx):
        size = len(gradient.shape)
        if size > 1:
            grads[idx] += tf.reduce_mean(-gradient, axis=tuple(range(1, size)), keepdims=True)
        def true_fn():
            s = tf.math.reduce_std(grads[idx]) + 1e-8
            grads[idx] = grads[idx] / s
        def false_fn():
            pass
        tf.cond(tf.size(gradient) > 2, true_fn, false_fn)
        gradient = grads[idx]
        return gradient
    
    def apply_orthogonal_gradients(self, params, grads, eps = 1e-16):
        for p, g in zip(params, grads):
            if tf.keras.backend.is_sparse(g):
                continue
            
            original_shape = g.shape
            w = tf.reshape(p, [-1])
            g = tf.reshape(g, [-1])

            proj = tf.tensordot(w, g, axes=1) / (tf.tensordot(w, w, axes=1) + eps)
            g_ortho = tf.cast(g, tf.float32) - proj * w
            g_norm = tf.norm(g)
            g_ortho_norm = tf.norm(g_ortho)
            g_ortho_scaled = g_ortho * (g_norm / (g_ortho_norm + eps))
            
            grads[self._get_variable_index(p)] = tf.reshape(g_ortho_scaled, original_shape)
    
    def apply_weight_decay(self, variable, gradient, lr):
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        elif self.weight_decay > 0.0:
            gradient += variable * self.weight_decay
    
    def accumulate_numerator(self, s, gradient, de_nom, d_lr, idx):
        if self.sn:
            size = tf.size(gradient)
            s = tf.reshape(s, (size // self.subset_size_[idx], self.subset_size_[idx]))
        flat_grad = tf.reshape(gradient, [-1])
        flat_div = tf.reshape(tf.divide(s, de_nom), [-1])
        dot_val = tf.tensordot(flat_grad, flat_div, axes=1)
        self.numerator_acc.assign_add(tf.cast(d_lr * dot_val, tf.float32))
    
    def get_second_moment_update(self, gradient, idx):
        size = tf.size(gradient)
        reshaped_grad = tf.reshape(gradient, (size // self.subset_size_[idx], self.subset_size_[idx]))
        second_moment_update = tf.reduce_sum(reshaped_grad ** 2, axis=1, keepdims=True)
        return second_moment_update
    
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
                size = tf.size(p)
                if self.sn:
                    reshaped_h_z = tf.reshape(h_z, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    reshaped_z = tf.reshape(z, (size // self.subset_size_[self._get_variable_index(p)], self.subset_size_[self._get_variable_index(p)]))
                    hessian_update = tf.reduce_sum(reshaped_h_z * reshaped_z, axis=1, keepdims=True)
                else:
                    hessian_update = h_z * z
                self.hessian[self._get_variable_index(p)].assign_add(hessian_update * alpha / num_samples)
    
    def update_hessian_moment(self, hessian_moment, step, idx):
        def true_fn2():
            hessian_moment.assign(hessian_moment * self.beta2 + self.hessian[idx] * (1.0 - self.beta2))
        def false_fn2():
            pass
        tf.cond(step % self.update_period == 0, true_fn2, false_fn2)
    
    def lookahead_merge(self, variable, step):
        def true_fn():
            slow_p = self.slow_momentum[self._get_variable_index(variable)]
            slow_p.assign(slow_p + self.lookahead_blending_alpha * (variable - slow_p))
            variable.assign(slow_p)
        
        def false_fn():
            pass
    
        tf.cond(step % self.lookahead_merge_time == 0, true_fn, false_fn)
    
    def apply_trust_ratio(self, variable, update):
        w_norm = tf.norm(variable, ord=2)
        g_norm = tf.norm(update, ord=2)
        trust_ratio = w_norm / g_norm
        trust_ratio = tf.where(
            w_norm > 0,
            tf.where(g_norm > 0, trust_ratio, 1.0),
            1.0,
        )
        if self.trust_clip:
            trust_ratio = tf.minimum(trust_ratio, 1.0)
        update *= trust_ratio
        return update
            
    def apply_cautious(self, update, gradient):
        mask = tf.cast(tf.math.greater(update * gradient, 0), gradient.dtype)
        numel = tf.cast(tf.size(mask), gradient.dtype)
        factor = numel / (tf.reduce_sum(mask) + 1)
        mask = mask * factor
        update = update * mask
        return update
    
    def apply_pnm(self, gradient, step, idx):
        noise_norm = math.sqrt((1 + self.beta2) ** 2 + self.beta2 ** 2)
        def true_fn():
            return self.pos_momentum[idx], self.neg_momentum[idx]
        def false_fn():
            return self.neg_momentum[idx], self.pos_momentum[idx]
        pos_momentum, neg_momentum = tf.cond(step % 2 == 1, true_fn, false_fn)
        pos_momentum.assign(pos_momentum * self.beta1 ** 2 + gradient * (1.0 - self.beta1 ** 2))
        return (pos_momentum  * 2.0 + neg_momentum * -1.0) * (1.0 / noise_norm)

    def update_step(self, gradient, variable, learning_rate):
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars, tape=None):
        self.tape = tape
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        # Return iterations for compat with tf.keras.
        return self._iterations

    def apply(self, grads, trainable_variables=None):
        """Update traininable variables according to provided gradient values.

        `grads` should be a list of gradient tensors
        with 1:1 mapping to the list of variables the optimizer was built with.

        `trainable_variables` can be provided
        on the first call to build the optimizer.
        """
        if len(grads) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return

        if trainable_variables is None:
            if not self.built:
                raise ValueError(
                    "When passing `grads` without `variables`, the optimizer "
                    "must already be built on a list of variables. "
                    "Call `optimizer.build(trainable_variables)` first. "
                )
            if len(grads) != len(self._trainable_variables_indices):
                raise ValueError(
                    "When passing `grads` as a list of gradient tensors, the "
                    f"gradients must match `optimizer.variables` one-to-on. "
                    f"Received a list of {len(grads)} gradients, but the "
                    f"optimizer is tracking {len(self._trainable_variables)} "
                    "trainable variables."
                )
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            # Optionally build optimizer.
            if not self.built:
                with backend.name_scope(self.name, caller=self):
                    self.build(trainable_variables)
                self.built = True
            self._check_variables_are_known(trainable_variables)

        with backend.name_scope(self.name, caller=self):
            # Filter empty gradients.
            grads, trainable_variables = self._filter_empty_gradients(
                grads, trainable_variables
            )

            # Overwrite targeted variables directly with their gradients if
            # their `overwrite_with_gradient` is set.
            grads, trainable_variables = (
                self._overwrite_variables_directly_with_gradients(
                    grads, trainable_variables
                )
            )

            if len(list(grads)) > 0:
                # Unscale gradients.
                scale = self.loss_scale_factor
                if scale is not None:
                    grads = [g if g is None else g / scale for g in grads]

                # Apply gradient updates.
                self._backend_apply_gradients(grads, trainable_variables)
                # Apply variable constraints after applying gradients.
                for variable in trainable_variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))

        # Update iteration counter.
        self._iterations.assign_add(1)

    def _backend_apply_gradients(self, grads, trainable_variables):
        """Apply method that can be overridden by different backends.

        JAX overrides it in order to deal with statelessness in gradient
        accumulation and EMA handling.

        The below implementation is intended to be generally backend-agnostic,
        but may not work with all backends.

        This method does 4 things:
        - Call the optimizer's update_step() to update trainable variables
            and optimizer variables.
        - Update EMA variables, if EMA is configured.
        - Update gradient accumulators, if gradient accumulation is configured.
        - Update the iteration counter.
        """
        if self.gradient_accumulation_steps:
            is_update_step = (
                self._iterations + 1
            ) % self.gradient_accumulation_steps == 0
            # `trainable_variables` might have been filtered in previous
            # processing steps, so we need to ensure the correct mapping between
            # `self._accumulated_gradients` and `trainable_variables`
            acc_grads = [
                self._accumulated_gradients[self._get_variable_index(v)]
                for v in trainable_variables
            ]

            def _update_step_fn(grads, trainable_variables):
                # Run update step with accumulated grads + reset accumulators
                steps = self.gradient_accumulation_steps
                grads = [
                    (g + acc_g) / steps for g, acc_g in zip(grads, acc_grads)
                ]

                # Apply clipping and weight decay.
                grads = self._clip_gradients(grads)
                self._apply_weight_decay(trainable_variables)

                self._backend_update_step(
                    grads, trainable_variables, self.learning_rate
                )
                self._backend_reset_gradient_accumulators()

            ops.cond(
                is_update_step,
                lambda: _update_step_fn(grads, trainable_variables),
                lambda: self._backend_increment_gradient_accumulators(
                    grads, acc_grads
                ),
            )
        else:
            # Apply clipping and weight decay.
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)

            # Run update step.
            self._backend_update_step(
                grads, trainable_variables, self.learning_rate
            )

        if self.use_ema:
            self._update_model_variables_moving_average(
                self._trainable_variables
            )
            if self.ema_overwrite_frequency:
                # Only when self.ema_overwrite_frequency is not None, we
                # overwrite the model variables.
                should_overwrite_model_vars = (
                    self.iterations + 1
                ) % self.ema_overwrite_frequency == 0
                ops.cond(
                    should_overwrite_model_vars,
                    lambda: self._overwrite_model_variables_with_average_value(
                        self._trainable_variables
                    ),
                    lambda: None,
                )

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.

        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        if self.sophia:
            def true_fn1():
                self.compute_hutchinson_hessian(
                    grads,
                    num_samples=self.num_samples,
                    distribution=self.distribution,
                )
            def false_fn1():
                pass
            tf.cond(self.iterations % self.update_period == 0, true_fn1, false_fn1)
        if self.orthograd:
            self.apply_orthogonal_gradients(trainable_variables, grads)
        self.update_step(grads, trainable_variables, learning_rate)

    def _backend_reset_gradient_accumulators(self):
        for g_acc in self._accumulated_gradients:
            if g_acc is not None:
                g_acc.assign(ops.zeros(g_acc.shape, dtype=g_acc.dtype))

    def _backend_increment_gradient_accumulators(self, grads, acc_grads):
        new_g_accs = [(g + acc_g) for g, acc_g in zip(grads, acc_grads)]
        for n_g_acc, g_acc in zip(new_g_accs, acc_grads):
            g_acc.assign(n_g_acc)

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        """Stateless version of `apply` that returns modified variables.

        Args:
            optimizer_variables: list of tensors containing the current values
                for the optimizer variables. These are native tensors and not
                `keras.Variable`s.
            grads: list of gradients to apply.
            trainable_variables: list of tensors containing the current values
                for the model variables. These are native tensors and not
                `keras.Variable`s.

        Returns: A tuple containing two list of tensors, the updated
            `trainable_variables` and the updated `optimizer_variables`.
        """
        self._check_super_called()

        if not self.built:
            raise ValueError(
                f"To call `stateless_apply`, {self.__class__.__name__} "
                "must be built (i.e. its variables must have been created). "
                "You can build it via `optimizer.build(trainable_variables)`."
            )
        if len(optimizer_variables) != len(self.variables):
            raise ValueError(
                "Argument `optimizer_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(optimizer_variables)}, but "
                f"expected {len(self.variables)} variables."
            )
        if len(trainable_variables) != len(self._trainable_variables):
            raise ValueError(
                "Argument `optimizer_variables` must be a list of tensors "
                "corresponding 1:1 to the trainable variables list that "
                "the optimizer was built with. Received "
                f"len(trainable_variables) == {len(trainable_variables)} "
                "whereas the optimizer was built with "
                f"{len(self._trainable_variables)} variables."
            )

        # Gather variable mapping
        mapping = list(
            zip(self._trainable_variables, trainable_variables)
        ) + list(zip(self.variables, optimizer_variables))

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.apply(grads)

        # Gather updated variables
        trainable_variables = []
        for v in self._trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                trainable_variables.append(new_v)
            else:
                trainable_variables.append(v)
        optimizer_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                optimizer_variables.append(new_v)
            else:
                optimizer_variables.append(v)
        return trainable_variables, optimizer_variables

    def scale_loss(self, loss):
        """Scale the loss before computing gradients.

        Scales the loss before gradients are computed in a `train_step`. This
        is primarily useful during mixed precision training to prevent numeric
        underflow.
        """
        if self.loss_scale_factor is not None:
            return loss * self.loss_scale_factor
        return loss

    @property
    def learning_rate(self):
        return self._get_current_learning_rate()

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if isinstance(self._learning_rate, backend.Variable):
            prev_lr_var = self._learning_rate
        else:
            prev_lr_var = None
        if isinstance(
            learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if isinstance(
                self._learning_rate, learning_rate_schedule.LearningRateSchedule
            ):
                raise TypeError(
                    "This optimizer was created with a `LearningRateSchedule`"
                    " object as its `learning_rate` constructor argument, "
                    "hence its learning rate is not settable. If you need the"
                    " learning rate to be settable, you should instantiate "
                    "the optimizer with a float `learning_rate` argument."
                )
            self._learning_rate.assign(learning_rate)
        if prev_lr_var is not None and not isinstance(
            self._learning_rate, backend.Variable
        ):
            # Untrack learning rate variable
            self._untrack_variable(prev_lr_var)

    def set_weights(self, weights):
        """Set the weights of the optimizer."""
        if not self.built:
            raise ValueError(
                "You are calling `set_weights()` on an optimizer that has not "
                "yet been built. Please call "
                "`optimizer.build(trainable_variables)` to create the "
                "optimizer weights before calling `set_weights()`."
            )
        for variable, weight in zip(self._variables, weights):
            if variable.shape != weight.shape:
                raise ValueError(
                    f"Optimizer variable {self._var_key(variable)} has shape "
                    f"{str(variable.shape)} not compatible with provided "
                    f"weight shape {str(weight.shape)}."
                )
            variable.assign(weight)

    def save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self.variables):
            store[str(i)] = variable.numpy()

    def load_own_variables(self, store):
        """Set the state of this optimizer object."""
        if len(store.keys()) != len(self.variables):
            msg = (
                f"Skipping variable loading for optimizer '{self.name}', "
                f"because it has {len(self.variables)} variables whereas "
                f"the saved optimizer has {len(store.keys())} variables. "
            )
            if len(self.variables) == 0:
                msg += (
                    "This is likely because the optimizer has not been "
                    "called/built yet."
                )
            warnings.warn(msg, stacklevel=2)
            return
        for i, variable in enumerate(self.variables):
            variable.assign(store[str(i)])

    def _get_current_learning_rate(self):
        if isinstance(
            self._learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            return self._learning_rate(self._iterations)
        elif isinstance(self._learning_rate, backend.Variable):
            return self._learning_rate
        elif callable(self._learning_rate):
            return self._learning_rate()
        return self._learning_rate

    def _overwrite_variables_directly_with_gradients(self, grads, vars):
        """Overwrite the variables directly by their gradients.

        This method is designed for a special case where we want to overwrite
        the variable directly with its computed gradient. For example, in float8
        training, new `scale` and `amax_history` are computed as gradients, and
        we want to overwrite them directly instead of following the typical
        procedure such as gradient descent with a learning rate, gradient
        clipping and weight decaying.

        After the update, the processed pairs will be filtered out.
        """
        # Shortcut for `tf.Variable` because it doesn't have a
        # `overwrite_with_gradient` attr.
        if not any(self._overwrite_variable_with_gradient(v) for v in vars):
            return grads, vars

        # Shallow copies
        filtered_grads = list(grads)
        filtered_vars = list(vars)

        # Iterate from right to left for safe popping
        for i in range(len(filtered_grads) - 1, -1, -1):
            g, v = filtered_grads[i], filtered_vars[i]
            if self._overwrite_variable_with_gradient(v):
                if self.gradient_accumulation_steps:
                    # Utilize a stateless manner for JAX compatibility
                    steps = self.gradient_accumulation_steps
                    is_update_step = (self._iterations + 1) % steps == 0
                    acc_g = self._accumulated_gradients[
                        self._get_variable_index(v)
                    ]
                    # `ops.maximum` is utilized for gradient accumulation for
                    # `overwrite_with_gradient=True` variables
                    new_g_acc = ops.cond(
                        is_update_step,
                        lambda: ops.zeros(g.shape, dtype=g.dtype),
                        lambda: ops.maximum(g, acc_g),
                    )
                    new_g = ops.cond(
                        is_update_step,
                        lambda: ops.maximum(g, acc_g),
                        lambda: g,
                    )
                    new_v = ops.cond(
                        is_update_step, lambda: new_g, lambda: v.value
                    )
                    v.assign(new_v)
                    acc_g.assign(new_g_acc)
                else:
                    v.assign(g)
                filtered_grads.pop(i)
                filtered_vars.pop(i)
        return filtered_grads, filtered_vars

    def _filter_empty_gradients(self, grads, vars):
        filtered_grads = list(grads)
        filtered_vars = list(vars)
        missing_grad_vars = []

        # Iterate from right to left for safe popping
        for i in range(len(filtered_grads) - 1, -1, -1):
            if filtered_grads[i] is None:
                filtered_grads.pop(i)
                v = filtered_vars.pop(i)
                try:
                    missing_grad_vars.append(v.path)
                except AttributeError:
                    # `tf.Variable` doesn't have `path` attr.
                    missing_grad_vars.append(v.name)

        if not filtered_grads:
            raise ValueError("No gradients provided for any variable.")
        if missing_grad_vars:
            warnings.warn(
                "Gradients do not exist for variables "
                f"{list(reversed(missing_grad_vars))} when minimizing the loss."
                " If using `model.compile()`, did you forget to provide a "
                "`loss` argument?"
            )
        return filtered_grads, filtered_vars

    def _clip_gradients(self, grads):
        if self.clipnorm and self.clipnorm > 0:
            return [
                self._clip_by_norm(g) if g is not None else g for g in grads
            ]
        elif self.global_clipnorm and self.global_clipnorm > 0:
            return clip_by_global_norm(grads, self.global_clipnorm)
        elif self.clipvalue and self.clipvalue > 0:
            v = self.clipvalue
            return [ops.clip(g, -v, v) if g is not None else g for g in grads]
        else:
            return grads

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decay.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
        if hasattr(self, "_built") and self._built:
            raise ValueError(
                "`exclude_from_weight_decay()` can only be configured before "
                "the optimizer is built."
            )

        # Use a `set` for the ids of `var_list` to speed up the searching
        if var_list:
            self._exclude_from_weight_decay = set(
                self._var_key(variable) for variable in var_list
            )
        else:
            self._exclude_from_weight_decay = set()

        # Precompile the pattern for `var_names` to speed up the searching
        if var_names and len(var_names) > 0:
            self._exclude_from_weight_decay_pattern = re.compile(
                "|".join(set(var_names))
            )
        else:
            self._exclude_from_weight_decay_pattern = None

        # Reset cache
        self._exclude_from_weight_decay_cache = dict()

    def _use_weight_decay(self, variable):
        variable_id = self._var_key(variable)

        # Immediately return the value if `variable_id` hits the cache
        if not hasattr(self, "_exclude_from_weight_decay_cache"):
            self._exclude_from_weight_decay_cache = dict()
        if variable_id in self._exclude_from_weight_decay_cache:
            return self._exclude_from_weight_decay_cache[variable_id]

        # Determine whether the variable should apply weight decay or not
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", set()
        )
        exclude_from_weight_decay_pattern = getattr(
            self, "_exclude_from_weight_decay_pattern", None
        )
        if variable_id in exclude_from_weight_decay:
            self._exclude_from_weight_decay_cache[variable_id] = False
            return False
        if exclude_from_weight_decay_pattern is not None:
            if (
                re.search(exclude_from_weight_decay_pattern, variable.name)
                is not None
            ):
                self._exclude_from_weight_decay_cache[variable_id] = False
                return False
        self._exclude_from_weight_decay_cache[variable_id] = True
        return True

    def _apply_weight_decay(self, variables):
        pass

    def _check_super_called(self):
        if not hasattr(self, "_lock"):
            raise RuntimeError(
                f"In optimizer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. "
                "Go add it!"
            )

    def _update_model_variables_moving_average(self, trainable_variables):
        """Update the stored moving average using the latest value."""
        if self.use_ema:
            for var, average in zip(
                trainable_variables, self._model_variables_moving_average
            ):
                if average is not None:
                    not_first_step = ops.not_equal(self.iterations, 0)
                    momentum = ops.multiply(
                        ops.cast(not_first_step, var.dtype), self.ema_momentum
                    )
                    average.assign(
                        ops.add(
                            ops.multiply(momentum, average),
                            ops.multiply(ops.subtract(1, momentum), var),
                        )
                    )

    def _overwrite_model_variables_with_average_value(
        self, trainable_variables
    ):
        """Overwrite model variables with its moving average."""
        if len(trainable_variables) != len(
            self._model_variables_moving_average
        ):
            raise ValueError(
                f"The length of model variables ({len(trainable_variables)}) "
                "to override does not match the length of model variables "
                "stored in the optimizer "
                f"({len(self._model_variables_moving_average)}). Please "
                "check if the optimizer was called on your model."
            )
        for var, average_var in zip(
            trainable_variables, self._model_variables_moving_average
        ):
            if average_var is not None:
                var.assign(average_var)

    def finalize_variable_values(self, var_list):
        """Set the final value of model's trainable variables.

        Sometimes there are some extra steps before ending the variable updates,
        such as overriding the model variables with its average value.

        Args:
          var_list: list of model variables.
        """
        if self.use_ema:
            # If the optimizer uses EMA, then when finalizing, we replace the
            # model variable value with its moving average stored inside
            # optimizer.
            self._overwrite_model_variables_with_average_value(var_list)

    def _obj_type(self):
        return "Optimizer"

    def get_config(self):
        """Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.

        Subclass optimizer should override this method to include other
        hyperparameters.

        Returns:
            Python dictionary.
        """

        if isinstance(
            self._learning_rate, learning_rate_schedule.LearningRateSchedule
        ):
            learning_rate = learning_rate_schedule.serialize(
                self._learning_rate
            )
        elif isinstance(self._learning_rate, backend.Variable):
            learning_rate = float(self._learning_rate.numpy())
        elif ops.is_tensor(self._learning_rate):
            learning_rate = float(self._learning_rate)
        elif callable(self._learning_rate):
            learning_rate = serialization_lib.serialize_keras_object(
                self._learning_rate
            )
        else:
            learning_rate = 0.5

        config = {
            "name": self.name,
            "learning_rate": learning_rate,
            "weight_decay": self.weight_decay,
            "clipnorm": self.clipnorm,
            "global_clipnorm": self.global_clipnorm,
            "clipvalue": self.clipvalue,
            "use_ema": self.use_ema,
            "ema_momentum": self.ema_momentum,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "loss_scale_factor": self.loss_scale_factor,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates an optimizer from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same optimizer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of get_config.
            custom_objects: A Python dictionary mapping names to additional
              user-defined Python objects needed to recreate this optimizer.

        Returns:
            An optimizer instance.
        """
        if "learning_rate" in config:
            if isinstance(config["learning_rate"], dict):
                config["learning_rate"] = (
                    serialization_lib.deserialize_keras_object(
                        config["learning_rate"], custom_objects=custom_objects
                    )
                )
        return cls(**config)

    def __setattr__(self, name, value):
        # Prevent users from attaching state to the
        # layer before `super()` is called -- since that
        # state would silently not be tracked.
        if name != "_lock":
            self._check_super_called()
        # Track Variables.
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def _clip_by_norm(self, values, axes=None):
        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = ops.sum(ops.square(values), axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = ops.where(pred, l2sum, ops.ones_like(l2sum))
        l2norm = ops.where(pred, ops.sqrt(l2sum_safe), l2sum)
        intermediate = ops.multiply(values, self.clipnorm)
        values_clip = ops.convert_to_tensor(intermediate) / ops.maximum(
            l2norm, self.clipnorm
        )
        return values_clip

    def _untrack_variable(self, variable):
        previous_lock_state = self._tracker.locked
        self._tracker.unlock()
        self._tracker.untrack(variable)
        if previous_lock_state is True:
            self._tracker.lock()


base_optimizer_keyword_args = """name: String. The name to use
            for momentum accumulator weights created by
            the optimizer.
        weight_decay: Float. If set, weight decay is applied.
        clipnorm: Float. If set, the gradient of each weight is individually
            clipped so that its norm is no higher than this value.
        clipvalue: Float. If set, the gradient of each weight is clipped to be
            no higher than this value.
        global_clipnorm: Float. If set, the gradient of all weights is clipped
            so that their global norm is no higher than this value.
        use_ema: Boolean, defaults to `False`.
            If `True`, exponential moving average
            (EMA) is applied. EMA consists of computing an exponential moving
            average of the weights of the model (as the weight values change
            after each training batch), and periodically overwriting the
            weights with their moving average.
        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.
            This is the momentum to use when computing
            the EMA of the model's weights:
            `new_average = ema_momentum * old_average + (1 - ema_momentum) *
            current_variable_value`.
        ema_overwrite_frequency: Int or None, defaults to None. Only used if
            `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
            we overwrite the model variable by its moving average.
            If None, the optimizer
            does not overwrite model variables in the middle of training,
            and you need to explicitly overwrite the variables
            at the end of training by calling
            `optimizer.finalize_variable_values()` (which updates the model
            variables in-place). When using the built-in `fit()` training loop,
            this happens automatically after the last epoch,
            and you don't need to do anything.
        loss_scale_factor: Float or `None`. If a float, the scale factor will
            be multiplied the loss before computing gradients, and the inverse
            of the scale factor will be multiplied by the gradients before
            updating variables. Useful for preventing underflow during
            mixed precision training. Alternately,
            `keras.optimizers.LossScaleOptimizer` will
            automatically set a loss scale factor.
        gradient_accumulation_steps: Int or `None`. If an int, model & optimizer
            variables will not be updated at every step; instead they will be
            updated every `gradient_accumulation_steps` steps, using the average
            value of the gradients since the last update. This is known as
            "gradient accumulation". This can be useful
            when your batch size is very small, in order to reduce gradient
            noise at each update step. EMA frequency will look at "accumulated"
            iterations value (optimizer steps // gradient_accumulation_steps).
            Learning rate schedules will look at "real" iterations value
            (optimizer steps).
"""


def global_norm(value_list):
    """Computes the global norm of multiple tensors."""
    squared_norms = [
        ops.sum(ops.square(v)) for v in value_list if v is not None
    ]
    squared_norm = ops.sum(ops.stack(squared_norms))
    return ops.sqrt(squared_norm)


def clip_by_global_norm(value_list, clip_norm):
    use_norm = global_norm(value_list)
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale_for_finite = clip_norm * ops.minimum(1.0 / use_norm, 1.0 / clip_norm)
    # If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
    # this will make scale NaN.
    scale = scale_for_finite + (use_norm - use_norm)
    return [v * scale if v is not None else v for v in value_list]
