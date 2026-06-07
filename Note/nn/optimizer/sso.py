import tensorflow as tf
from Note.nn.optimizer import optimizer
from typing import Optional, Tuple, Union


def power_iteration(w: tf.Tensor, steps: int = 50) -> Tuple[tf.Tensor, tf.Tensor]:
    """Leading singular triplet (sigma, u, v) via bilateral power iteration (bf16)."""
    w = tf.cast(w, tf.bfloat16)
    w_shape = tf.shape(w)
    v = tf.ones([w_shape[1], 1], dtype=tf.bfloat16)

    for _ in range(steps):
        tmp = tf.matmul(tf.transpose(w), tf.matmul(w, v))
        v = tf.nn.l2_normalize(tmp, axis=0)

    u = tf.matmul(w, v)
    u = tf.nn.l2_normalize(u, axis=0)
    return u, v


def msign(x: tf.Tensor, steps: int) -> tf.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
    x_shape = tf.shape(x)
    transpose_flag = tf.greater(x_shape[0], x_shape[1])

    x = tf.cond(transpose_flag, lambda: tf.transpose(x), lambda: x)
    fro_norm = tf.linalg.norm(x, ord="fro")
    x = x / tf.maximum(fro_norm, tf.constant(1e-7, dtype=x.dtype))
    x = tf.cast(x, tf.bfloat16)

    coefficients = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]

    for i in range(steps):
        coef_a, coef_b, coef_c = coefficients[i] if i < 8 else coefficients[-1]
        a = tf.matmul(x, tf.transpose(x))
        aa = tf.matmul(a, a)
        b = coef_b * a + coef_c * aa
        x = coef_a * x + tf.matmul(b, x)

    x = tf.cond(transpose_flag, lambda: tf.transpose(x), lambda: x)
    return x


def compute_f_tensor(
    x: tf.Tensor,
    theta: tf.Tensor,
    lambda_value: Union[float, tf.Tensor],
    msign_steps: int = 8,
) -> tf.Tensor:
    """f(lambda) = <Θ, msign(G + lambdaΘ)>. Returns 0-d tensor."""
    z = x + lambda_value * theta
    phi = msign(z, steps=msign_steps)
    return tf.reduce_sum(theta * phi)


def find_bracket(
    x: tf.Tensor,
    theta: tf.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    max_expansions: int = 10,
    msign_steps: int = 8,
    tolerance_f: float = 1e-8,
) -> Tuple[Optional[float], Optional[float], tf.Tensor, tf.Tensor]:
    """Find lambda_l < lambda_r such that f(lambda_l) <= 0 <= f(lambda_r)."""
    lambda_0 = initial_guess
    f0 = compute_f_tensor(x, theta, lambda_0, msign_steps)

    if tf.abs(f0) < tolerance_f:
        return lambda_0, lambda_0, f0, f0

    step = initial_step if f0 < 0 else -initial_step
    lambda_prev = lambda_0
    f_prev = f0

    for _ in range(max_expansions):
        lambda_new = lambda_prev + step
        f_new = compute_f_tensor(x, theta, lambda_new, msign_steps)

        sign_prev = f_prev <= 0.0
        sign_new = f_new <= 0.0

        if sign_prev != sign_new:
            if f_prev <= 0 <= f_new:
                lambda_l, f_l = lambda_prev, f_prev
                lambda_r, f_r = lambda_new, f_new
            elif f_new <= 0 <= f_prev:
                lambda_l, f_l = lambda_new, f_new
                lambda_r, f_r = lambda_prev, f_prev
            elif abs(f_prev) <= abs(f_new):
                lambda_l = lambda_r = lambda_prev
                f_l = f_r = f_prev
            else:
                lambda_l = lambda_r = lambda_new
                f_l = f_r = f_new
            return lambda_l, lambda_r, f_l, f_r

        step *= 2.0
        lambda_prev, f_prev = lambda_new, f_new

    return None, None, f0, f0


def solve_lambda_with_bisection(
    x: tf.Tensor,
    theta: tf.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 8,
) -> float:
    """Solve lambda such that f(lambda) = 0 using bisection."""
    lambda_l, lambda_r, f_l, f_r = find_bracket(
        x,
        theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )
    if lambda_l is None or lambda_r is None:
        return 0.0

    if tf.abs(f_l) < tf.abs(f_r):
        best_lambda, best_f = lambda_l, f_l
    else:
        best_lambda, best_f = lambda_r, f_r

    if tf.abs(best_f) <= tolerance_f:
        return best_lambda

    for _ in range(1, max_iterations + 1):
        lambda_mid = 0.5 * (lambda_l + lambda_r)
        f_mid = compute_f_tensor(x, theta, lambda_mid, msign_steps)

        if tf.abs(f_mid) < tf.abs(best_f):
            best_lambda, best_f = lambda_mid, f_mid

        if tf.abs(f_mid) <= tolerance_f:
            return lambda_mid

        if f_mid < 0:
            lambda_l, f_l = lambda_mid, f_mid
        else:
            lambda_r, f_r = lambda_mid, f_mid

    return best_lambda


def compute_spectral_ball_update(
    weight: tf.Tensor,
    momentum: tf.Tensor,
    power_iteration_steps: int,
    msign_steps: int,
    solver_tolerance_f: float,
    solver_max_iterations: int,
) -> tf.Tensor:
    """Compute spectral ball constrained update direction."""
    momentum_fp32 = tf.cast(momentum, tf.float32)
    norm = tf.linalg.norm(momentum_fp32, ord="fro")
    momentum_fp32 = momentum_fp32 / tf.maximum(norm, tf.constant(1e-8, dtype=tf.float32))

    u, v = power_iteration(weight, steps=power_iteration_steps)
    theta = tf.matmul(u, tf.transpose(v))

    lambda_value = solve_lambda_with_bisection(
        momentum_fp32,
        theta=theta,
        initial_guess=0.0,
        initial_step=1e-3,
        tolerance_f=solver_tolerance_f,
        max_iterations=solver_max_iterations,
        max_expansions=10,
        msign_steps=msign_steps,
    )

    z = momentum_fp32 + lambda_value * theta
    return msign(z, steps=msign_steps)


class SpectralSphere(optimizer.Optimizer):
    """Controlled LLM Training on Spectral Sphere.

    This optimizer constrains weight matrices to lie on a spectral sphere of fixed
    radius (implicit via the retraction / sign-function projection). The optimization
    proceeds by:

    1. Power iteration to compute spectral norm and top singular vectors (u, v)
    2. Form Θ = uvᵀ
    3. Solve for Lagrange multiplier lambda: <Θ, msign(M + lambdaΘ)> = 0
    4. Compute update direction Φ = msign(M + lambdaΘ)
    5. Update: W ← W - lr * Φ

    References:
        - Spectral MuP: Spectral Control of Feature Learning
        - Modular Duality in Deep Learning. arXiv:2410.21265 (2024).

    Args:
        learning_rate (float): Learning rate (default 3e-4).
        momentum (float): Momentum for the internal SGD (default 0.9).
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Use decoupled weight decay (AdamW style).
        nesterov (bool): Use Nesterov momentum.
        power_iteration_steps (int): Number of power iteration steps.
        msign_steps (int): Number of Newton-Schulz iterations for matrix sign.
        solver_tolerance_f (float): Tolerance for the lambda solver.
        solver_max_iterations (int): Maximum iterations for the bisection solver.
        maximize (bool): Maximize the objective instead of minimizing.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        nesterov: bool = True,
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        solver_tolerance_f: float = 1e-8,
        solver_max_iterations: int = 100,
        maximize: bool = False,
        name: str = "spectralsphere",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.nesterov = nesterov
        self.power_iteration_steps = power_iteration_steps
        self.msign_steps = msign_steps
        self.solver_tolerance_f = solver_tolerance_f
        self.solver_max_iterations = solver_max_iterations
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        for var in var_list:
            self.momentum_buffer.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError("SpectralSphere does not support sparse gradients.")

        if variable.dtype.is_complex:
            raise RuntimeError("SpectralSphere does not support complex parameters.")

        if len(variable.shape) != 2:
            raise ValueError(f"{self.name} only supports 2D parameters.")

        if self.maximize:
            gradient = -gradient

        lr = tf.cast(learning_rate, variable.dtype)

        gradient = self.apply_weight_decay(variable, gradient, lr)

        idx = self._get_variable_index(variable)
        buf = self.momentum_buffer[idx]
        buf.assign(buf * self.momentum + gradient * (1.0 - self.momentum))

        if self.nesterov:
            update = gradient * (1.0 - self.momentum) + buf * self.momentum
        else:
            update = buf

        update = compute_spectral_ball_update(
            weight=variable,
            momentum=update,
            power_iteration_steps=self.power_iteration_steps,
            msign_steps=self.msign_steps,
            solver_tolerance_f=self.solver_tolerance_f,
            solver_max_iterations=self.solver_max_iterations,
        )

        variable.assign_add(-lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "nesterov": self.nesterov,
                "power_iteration_steps": self.power_iteration_steps,
                "msign_steps": self.msign_steps,
                "solver_tolerance_f": self.solver_tolerance_f,
                "solver_max_iterations": self.solver_max_iterations,
                "maximize": self.maximize,
            }
        )
        return config


class SpectralSphere_e(optimizer.Optimizer):
    """Controlled LLM Training on Spectral Sphere (enhanced version).

    This optimizer constrains weight matrices (2D only) to lie on a spectral
    sphere of fixed radius via retraction + matrix sign projection.
    The optimization proceeds by:

    1. Power iteration → spectral norm & top singular vectors (u, v)
    2. Form Θ = uvᵀ
    3. Solve for Lagrange multiplier λ: ⟨Θ, msign(M + λΘ)⟩ = 0
    4. Compute update direction Φ = msign(M + λΘ)
    5. Update: W ← W - lr * Φ

    Enhanced with all features from the new BaseOptimizer framework:
    - Orthogonal gradients, AGC, Cautious updates, Lookahead, PNM momentum
    - Trust-ratio / layer-wise LR adaptation
    - Weight decay (decoupled/coupled)

    References:
        - Spectral MuP: Spectral Control of Feature Learning
        - Modular Duality in Deep Learning. arXiv:2410.21265 (2024).

    Args:
        learning_rate (float): Learning rate (default 3e-4).
        momentum (float): Momentum for internal SGD (default 0.9).
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Decoupled weight decay (AdamW style).
        fixed_decay (bool): Use fixed decay instead of lr-scaled.
        nesterov (bool): Nesterov momentum.
        power_iteration_steps (int): Power iteration steps for spectral norm.
        msign_steps (int): Newton-Schulz steps for matrix sign (Polar-Express).
        solver_tolerance_f (float): Lambda solver tolerance.
        solver_max_iterations (int): Max bisection iterations.
        maximize (bool): Maximize instead of minimize.
        orthograd (bool): Orthogonalize gradients.
        lookahead_merge_time (int): Lookahead merge interval.
        lookahead_blending_alpha (float): Lookahead blending factor.
        lookahead (bool): Enable Lookahead.
        pnm (bool): Use PNM (Positive-Negative Momentum).
        agc (bool): Adaptive Gradient Clipping.
        cautious (bool): Cautious update.
        trust_ratio (bool): Layer-wise trust ratio.
        trust_clip (bool): Clip trust ratio to 1.0.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        nesterov: bool = True,
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        solver_tolerance_f: float = 1e-8,
        solver_max_iterations: int = 100,
        maximize: bool = False,
        orthograd: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        lookahead: bool = False,
        pnm: bool = False,
        agc: bool = False,
        cautious: bool = False,
        trust_ratio: bool = False,
        trust_clip: bool = False,
        shampoo: bool = False,
        update_freq: int = 1,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="spectralsphere_e",
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
        self.momentum = momentum
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.nesterov = nesterov
        self.power_iteration_steps = power_iteration_steps
        self.msign_steps = msign_steps
        self.solver_tolerance_f = solver_tolerance_f
        self.solver_max_iterations = solver_max_iterations
        self.maximize = maximize

        # Extra features from BaseOptimizer framework
        self.orthograd = orthograd
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.lookahead = lookahead
        self.pnm = pnm
        self.agc = agc
        self.cautious = cautious
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.shampoo = shampoo
        self.update_freq = update_freq

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)

        self.momentum_buffer = []
        self.slow_momentum = []
        self.pos_momentum = []
        self.neg_momentum = []

        for var in var_list:
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(var))
                self._track_variable(self.slow_momentum[-1])

            if self.pnm:
                self.pos_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="pos_momentum"
                    )
                )
                self.neg_momentum.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="neg_momentum"
                    )
                )
            else:
                self.momentum_buffer.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="momentum_buffer"
                    )
                )

    def update_step(self, grads, trainable_variables, learning_rate):
        if self.orthograd:
            self.apply_orthogonal_gradients(trainable_variables, grads)

        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    "SpectralSphere_e does not support sparse gradients."
                )

            if p.dtype.is_complex:
                raise RuntimeError(
                    "SpectralSphere_e does not support complex parameters."
                )

            if len(p.shape) != 2:
                raise ValueError(f"{self.name} only supports 2D parameters.")

            grad = g
            if self.maximize:
                grad = -grad

            lr = tf.cast(learning_rate, p.dtype)
            step = tf.cast(self.iterations + 1, p.dtype)

            if self.agc:
                grads[self._get_variable_index(p)] = self.apply_agc(p, grad)
                grad = grads[self._get_variable_index(p)]

            grad = self.apply_weight_decay(p, grad, lr)

            idx = self._get_variable_index(p)
            if self.pnm:
                update = self.apply_pnm(grad, step, idx)
            else:
                buf = self.momentum_buffer[idx]
                buf.assign(buf * self.momentum + grad * (1.0 - self.momentum))

                if self.nesterov:
                    update = grad * (1.0 - self.momentum) + buf * self.momentum
                else:
                    update = buf
            
            if self.shampoo:
                for dim_id, dim in enumerate(grad.shape.as_list()):
                    precond = self.precond[self._get_variable_index(p)]["precond_{}".format(dim_id)]
                    inv_precond = self.inv_precond[self._get_variable_index(p)]["inv_precond_{}".format(dim_id)]
        
                    # mat_{dim_id}(grad)
                    current_rank = len(grad.shape)
                    perm = list(range(current_rank))
                    perm[0], perm[dim_id] = perm[dim_id], perm[0]
                    grad = tf.transpose(grad, perm=perm)
                    grad = tf.reshape(grad, (dim, -1))
                    
                    self.update_inv_precond(self, grad, precond, inv_precond)
                
                    if dim_id == 0:
                        update = tf.matmul(inv_precond, update)
                    else:
                        update = tf.matmul(update, inv_precond)

            update = compute_spectral_ball_update(
                weight=p,
                momentum=update,
                power_iteration_steps=self.power_iteration_steps,
                msign_steps=self.msign_steps,
                solver_tolerance_f=self.solver_tolerance_f,
                solver_max_iterations=self.solver_max_iterations,
            )

            if self.trust_ratio:
                update = self.apply_trust_ratio(p, update)

            if self.cautious:
                update = self.apply_cautious(update, grad)

            p.assign_add(-lr * update)

            if self.lookahead:
                self.lookahead_merge(p, step)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "weight_decouple": self.weight_decouple,
                "fixed_decay": self.fixed_decay,
                "nesterov": self.nesterov,
                "power_iteration_steps": self.power_iteration_steps,
                "msign_steps": self.msign_steps,
                "solver_tolerance_f": self.solver_tolerance_f,
                "solver_max_iterations": self.solver_max_iterations,
                "maximize": self.maximize,
                "orthograd": self.orthograd,
                "lookahead_merge_time": self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "lookahead": self.lookahead,
                "pnm": self.pnm,
                "agc": self.agc,
                "cautious": self.cautious,
                "trust_ratio": self.trust_ratio,
                "trust_clip": self.trust_clip,
                "shampoo": self.shampoo,
                "update_freq": self.update_freq,
            }
        )
        return config