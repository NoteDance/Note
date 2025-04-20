""" Kron
Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
import string
import numpy as np


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    max_prob_ = tf.convert_to_tensor(max_prob, dtype=tf.float32) 
    min_prob_ = tf.convert_to_tensor(min_prob, dtype=tf.float32)
    decay_ = tf.convert_to_tensor(decay, dtype=tf.float32)
    flat_start_ = tf.convert_to_tensor(flat_start, dtype=tf.float32)

    def _schedule(n):
        """Exponential anneal with flat start."""
        prob = max_prob_ * tf.exp(-decay_ * (n - flat_start_))
        prob = tf.clip_by_value(prob, clip_value_min=min_prob_, clip_value_max=max_prob_)
        return prob

    return _schedule


class Kron(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.0003,
        weight_decay=0.0,
        b1=0.9,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        precond_lr=0.1,
        precond_init_scale=1.0,
        mu_dtype=None,
        precond_dtype=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="kron",
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
        self.b1 = b1
        self.preconditioner_update_probability = preconditioner_update_probability
        self.max_size_triangular = max_size_triangular
        self.min_ndim_triangular = min_ndim_triangular
        self.memory_save_mode = memory_save_mode
        self.momentum_into_precond_update = momentum_into_precond_update
        self.precond_lr = precond_lr
        self.precond_init_scale = precond_init_scale
        self.mu_dtype = mu_dtype
        self.precond_dtype = precond_dtype
        
        self.rng = tf.random.Generator.from_seed(42)
        
        if preconditioner_update_probability is None:
            self.preconditioner_update_probability = precond_update_prob_schedule()

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.Q = []
        self.exprs = []
        if self.precond_dtype is None:
            self.precond_dtype = tf.float32
        self._prob_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._update_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._track_variable(self._prob_step)
        self._track_variable(self._update_counter)
        self.step = []
        for var in var_list:
            self.momentum_buffer.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                )
            )
            Q, exprs = _init_Q_exprs(
                var,
                self.precond_init_scale,
                self.max_size_triangular,
                self.min_ndim_triangular,
                self.memory_save_mode,
                dtype=self.precond_dtype,
            )
            self.Q.append(Q)
            self._track_variable(self.Q[-1])
            self.exprs.append(exprs)
            self.step.append(0)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        total_momentum_size = 0
        total_momentum_mb = 0
        total_precond_size = 0
        total_precond_mb = 0
        
        # update preconditioners all together deterministically
        update_prob = self.preconditioner_update_probability
        if callable(update_prob):
            update_prob = update_prob(tf.cast(self._prob_step, tf.float32))
        self._update_counter.assign_add(1)
        
        do_update = tf.cast(self._update_counter, update_prob.dtype) >= tf.cast(1, update_prob.dtype) / update_prob
        self._update_counter.assign(
            tf.cond(
                do_update,
                lambda: tf.constant(0, dtype=tf.int32),
                lambda: self._update_counter
            )
        )
        self._prob_step.assign_add(1)
        
        # balance preconditioners roughly every 100 updates
        balance = tf.logical_and(self.rng.uniform(shape=[], minval=0.0, maxval=1.0) < 0.01, do_update)
        
        if self.precond_dtype is None:
            self.precond_dtype = tf.float32
        
        momentum_size = tf.size(self.momentum_buffer[self._get_variable_index(variable)])
        dtype = self.momentum_buffer[self._get_variable_index(variable)].dtype
        itemsize = np.dtype(dtype if isinstance(dtype, str) else dtype.as_numpy_dtype).itemsize
        momentum_mb = momentum_size * itemsize / (2**20)
        total_momentum_size += momentum_size
        total_momentum_mb += momentum_mb

        precond_size = sum(tf.get_static_value(tf.size(q)) for q in self.Q[self._get_variable_index(variable)])
        precond_mb = sum(tf.get_static_value(tf.size(q)) * q.dtype.size for q in self.Q[self._get_variable_index(variable)]) / (2**20)
        total_precond_size += precond_size
        total_precond_mb += precond_mb
        
        self.step[self._get_variable_index(variable)] += 1
        
        momentum_buffer = self.momentum_buffer[self._get_variable_index(variable)]
        momentum_buffer.assign(momentum_buffer * self.b1 + gradient * (1 - self.b1))
        # restore momentum dtype
        if self.mu_dtype is not None:
            momentum_buffer = self.momentum_buffer[self._get_variable_index(variable)] = tf.cast(momentum_buffer, self.mu_dtype)
        debiased_momentum = momentum_buffer / (1 - self.b1 ** self.step[self._get_variable_index(variable)])
        debiased_momentum = tf.cast(debiased_momentum, self.precond_dtype)
        
        # balance preconditioners about every 100 updates
        def true_fn():
            _balance_Q(self.Q[self._get_variable_index(variable)])

        def false_fn():
            pass
        
        tf.cond(tf.logical_and(len(gradient.shape) > 1, balance), true_fn, false_fn)
        
        # update preconditioner
        def true_fn():
            _update_precond(
                self.Q[self._get_variable_index(variable)],
                self.exprs[self._get_variable_index(variable)],
                debiased_momentum if self.momentum_into_precond_update else tf.cast(gradient, self.precond_dtype),
                tf.convert_to_tensor(self.precond_lr, dtype=self.precond_dtype),
                tf.convert_to_tensor(tf.experimental.numpy.finfo(self.precond_dtype).tiny, dtype=self.precond_dtype),
                )
            
        def false_fn():
            pass
        
        tf.cond(do_update, true_fn, false_fn)
        
        # precondition gradients
        pre_grad = _precond_grad(self.Q[self._get_variable_index(variable)], self.exprs[self._get_variable_index(variable)], debiased_momentum)

        # clip update RMS
        pre_grad = _clip_update_rms(pre_grad)

        # apply weight decay and update parameters
        if self.weight_decay != 0 and len(variable.shape) >= 2:
            pre_grad += variable * self.weight_decay
        variable.assign_add(tf.cast(pre_grad, variable.dtype) * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "b1": self.b1,
                "preconditioner_update_probability": self.preconditioner_update_probability,
                "max_size_triangular": self.max_size_triangular,
                "min_ndim_triangular": self.min_ndim_triangular,
                "memory_save_mode": self.memory_save_mode,
                "momentum_into_precond_update": self.momentum_into_precond_update,
                "precond_lr": self.precond_lr,
                "precond_init_scale": self.precond_init_scale,
                "mu_dtype": self.mu_dtype,
                "precond_dtype": self.precond_dtype,
                "exprs": self.exprs,
                "step": [self.iterations.numpy() for _ in range(len(self.step))],
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


def _init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [tf.Variable(scale * tf.ones_like(t, dtype=dtype))]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "smart_one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            sorted_shape = sorted(shape)
            if len(shape) > 1 and sorted_shape[-1] > sorted_shape[-2]:
                dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'smart_one_diag', 'one_diag', 'all_diag']"
            )

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (
                size == 1
                or size > max_size
                or len(shape) < min_ndim_triangular
                or dim_d
            ):
                # use diagonal matrix as preconditioner for this dim
                Q.append(tf.Variable(scale * tf.ones(size, dtype=dtype)))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(tf.Variable(scale * tf.eye(size, dtype=dtype)))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


def _balance_Q(Q_in):
    norms = tf.stack([tf.norm(q, ord=np.inf) for q in Q_in])
    geometric_mean = tf.exp(tf.reduce_mean(tf.math.log(norms)))
    norms = geometric_mean / norms
    for i, q in enumerate(Q_in):
        q.assign(q * norms[i])


def _lb(A, max_abs):
    """Cheap lower bound for the spectral norm of A."""
    A /= max_abs
    a0 = tf.einsum("ij,ij->j", A, A)
    i = tf.argmax(a0)
    x = tf.reshape(tf.gather(A, indices=i, axis=1), [-1])
    x = tf.einsum("i,ij->j", x, A)
    x /= tf.norm(x)
    x = tf.einsum("j,kj->k", x, A)
    x = tf.norm(x)
    x *= max_abs
    return x


def solve_triangular_right(A, X):
    A_float = tf.cast(A, tf.float32)
    B = tf.reshape(X, [-1, tf.shape(X)[-1]])
    B_float = tf.cast(B, tf.float32)
    Z = tf.linalg.triangular_solve(tf.transpose(A_float),
                                   tf.transpose(B_float),
                                   lower=True)
    Y = tf.transpose(Z)
    return Y


def _solve_triangular_right(X, A):
    """X @ inv(A)"""
    orig_dtype = A.dtype
    return (
        tf.reshape(tf.cast(solve_triangular_right(
            A,
            X,
        )
        , orig_dtype)
        , X.shape)
    )


def _calc_A_and_conjB(exprA, G, Q):
    """Calculate A and conjB."""
    order = G.shape.ndims
    V = tf.random.normal(tf.shape(G), dtype=G.dtype)
    eps = tf.convert_to_tensor(tf.experimental.numpy.finfo(tf.float32).eps, dtype=G.dtype)
    G += tf.sqrt(eps) * tf.reduce_mean(tf.abs(G)) * V
    if order > 0:
        conjB = tf.transpose(V, perm=tf.concat([tf.range(1, order), [0]], axis=0))
        for i, q in enumerate(Q):
            if tf.rank(q) < 2:
                conjB = tf.divide(conjB, q)
            else:
                conjB = _solve_triangular_right(conjB, q)
            if i < order - 1:
                perm = list(range(order))
                perm[i], perm[order - 1] = perm[order - 1], perm[i]
                conjB = tf.transpose(conjB, perm=perm)
    else:
        conjB = V # For scalar, conjB is just V
    A = tf.einsum(exprA, *(Q + [G]))
    return A, conjB


def _update_precond(Q, exprs, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs
    A, conjB = _calc_A_and_conjB(exprA, G, Q)
    for i, (q, exprG) in enumerate(zip(Q, exprGs)):
        term1 = tf.einsum(exprG, A, A)
        term2 = tf.einsum(exprG, conjB, conjB)
        term1, term2 = term1 - term2, term1 + term2
        term1 *= step
        norm = tf.norm(term2, ord=np.inf)
        if len(q.shape) < 2:
            term1 *= q / tf.maximum(norm, tiny)
        else:
            term1 = tf.linalg.band_part(term1, 0, -1)
            term1 /= tf.maximum(tf.where(norm > 0, _lb(term2, norm), norm), tiny)
            term1 = tf.matmul(term1, q)
        q.assign(q - term1)


def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return tf.einsum(exprs[-1], *(Q + Q + [G]))


def _clip_update_rms(g):
    rms = tf.sqrt(tf.reduce_mean(tf.square(g))) + 1e-12
    factor = tf.minimum(tf.constant(1.0, dtype=g.dtype), 1.1 / rms)
    return g * factor