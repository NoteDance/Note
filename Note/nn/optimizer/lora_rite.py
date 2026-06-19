import tensorflow as tf
from Note.nn.optimizer import optimizer
from typing import Dict, List, Tuple


_FLOAT32_MAX = 3.4028235e38
_FLOAT32_MIN = -3.4028235e38


def _make_symmetric(tensor: tf.Tensor) -> tf.Tensor:
    """``(T + Tᵀ) / 2`` — symmetrize a square matrix."""
    return (tensor + tf.linalg.matrix_transpose(tensor)) * 0.5


def _nan_to_num(tensor: tf.Tensor) -> tf.Tensor:
    """Replace NaN→0, +Inf→float32 max, -Inf→float32 min (matches
    ``torch.nan_to_num`` default behaviour)."""
    dtype = tensor.dtype
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    tensor = tf.where(
        tf.math.logical_and(tf.math.is_inf(tensor), tensor > 0),
        tf.fill(tf.shape(tensor), tf.constant(_FLOAT32_MAX, dtype)),
        tensor,
    )
    tensor = tf.where(
        tf.math.logical_and(tf.math.is_inf(tensor), tensor < 0),
        tf.fill(tf.shape(tensor), tf.constant(_FLOAT32_MIN, dtype)),
        tensor,
    )
    return tensor


def _reduce_rms(tensor: tf.Tensor) -> tf.Tensor:
    """Root-mean-square of all elements, as a scalar."""
    return tf.sqrt(tf.reduce_mean(tf.square(tensor)))


def _bias_corrected_decay(step: tf.Tensor, decay: float) -> tf.Tensor:
    """Bias-corrected EMA decay rate for the given (0-indexed) ``step``.

    Matches ``decay * (1 - decay^step) / (1 - decay^(step+1))`` — yields a
    decay factor that compensates for zero-initialised moment buffers,
    similarly in spirit to Adam's bias correction but expressed as a
    per-step *decay rate* rather than a post-hoc divisor.
    """
    next_step = tf.cast(step, tf.float32) + 1.0
    decay_t = tf.constant(decay, dtype=tf.float32)
    numerator = 1.0 - tf.pow(decay_t, next_step - 1.0)
    denominator = 1.0 - tf.pow(decay_t, next_step)
    return decay_t * numerator / denominator


def _inverse_sqrt(
    tensor: tf.Tensor,
    escape: tf.Tensor,
    eps: float,
    eps_root: float,
    relative_epsilon: bool,
) -> tf.Tensor:
    """Symmetric inverse matrix square root via eigendecomposition.

    ``inverse_root = V * diag(1 / (sqrt(max(λ,0) + escape + eps_root) + eps)) * Vᵀ``
    """
    eigenvalues, eigenvectors = tf.linalg.eigh(_make_symmetric(tensor))
    if relative_epsilon:
        eps_root = tf.maximum(tf.reduce_max(eigenvalues), 0.0) * eps_root
    inverse_eigenvalues = 1.0 / (
        tf.sqrt(tf.maximum(eigenvalues, 0.0) + escape + eps_root) + eps
    )
    inverse_root = tf.matmul(
        eigenvectors * inverse_eigenvalues[tf.newaxis, :],
        eigenvectors, transpose_b=True,
    )
    return _make_symmetric(tf.cast(inverse_root, tensor.dtype))


def _transform_second_moment(moments: tf.Tensor, projection: tf.Tensor) -> tf.Tensor:
    """Rotate a second-moment (covariance-like) matrix into a new QR basis:
    ``P @ M @ Pᵀ``, NaN-safe and re-symmetrized."""
    result = tf.matmul(tf.matmul(projection, moments), projection, transpose_b=True)
    return _make_symmetric(_nan_to_num(result))


def _transform_first_moment(moments: tf.Tensor, projection: tf.Tensor) -> tf.Tensor:
    """Rotate a first-moment matrix into a new QR basis: ``M @ Pᵀ``."""
    return tf.matmul(moments, projection, transpose_b=True)


def _get_unmagnified_rotate_second_escape(
    new_moments: tf.Tensor, old_moments: tf.Tensor
) -> tf.Tensor:
    """RITE 'escape' correction term: the smaller of the largest eigenvalue
    drop and the trace drop caused by rotating into the new basis (both
    clamped to ≥ 0). Both matrices are assumed already symmetric."""
    old_eigs = tf.linalg.eigh(old_moments)[0]
    new_eigs = tf.linalg.eigh(new_moments)[0]
    zero = tf.constant(0.0, dtype=old_eigs.dtype)
    eigen_diff = tf.maximum(tf.reduce_max(old_eigs - new_eigs), zero)
    trace_diff = tf.maximum(
        tf.linalg.trace(old_moments) - tf.linalg.trace(new_moments), zero
    )
    return tf.minimum(eigen_diff, trace_diff)


def _get_preconditioned_update(
    grad: tf.Tensor,
    moments: tf.Tensor,
    escape: tf.Tensor,
    eps: float,
    eps_root: float,
    relative_epsilon: bool,
    apply_escape: bool,
) -> tf.Tensor:
    """``grad @ inverse_sqrt(moments)``, NaN-safe."""
    if not apply_escape:
        escape = tf.zeros((), dtype=moments.dtype)
    inverse_root = _inverse_sqrt(moments, escape, eps, eps_root, relative_epsilon)
    return _nan_to_num(tf.matmul(grad, inverse_root))


def _update_first_moment(
    step: tf.Tensor, update: tf.Tensor, moments: tf.Tensor, beta1: float
) -> tf.Tensor:
    decay = _bias_corrected_decay(step, beta1)
    return update * (1.0 - decay) + moments * decay


def _compute_second_moment(update: tf.Tensor) -> tf.Tensor:
    """``(Uᵀ U) / n_rows``, symmetrized."""
    n_rows = tf.cast(tf.shape(update)[0], update.dtype)
    return _make_symmetric(tf.matmul(update, update, transpose_a=True)) / n_rows


def _update_second_moment(
    step: tf.Tensor, update: tf.Tensor, moments: tf.Tensor, beta2: float
) -> tf.Tensor:
    decay = _bias_corrected_decay(step, beta2)
    return update * (1.0 - decay) + moments * decay


def _update_second_escape(
    step: tf.Tensor, update: tf.Tensor, moments: tf.Tensor, beta2: float
) -> tf.Tensor:
    decay = _bias_corrected_decay(step, beta2)
    return update * (1.0 - decay) + moments * decay


class LoRARite(optimizer.Optimizer):
    """Robust Invariant Transformation Equilibration for LoRA optimization.

    .. important::
        This optimizer expects LoRA factors in **strict alternating order**:
        ``lora_a_1, lora_b_1, lora_a_2, lora_b_2, ...``. Build the trainable
        variable list (or your model's parameter ordering) accordingly.
        Unpaired variables, or pairs where only one factor received a
        gradient on a given step, are skipped — matching common fine-tuning
        workflows where only part of the model may be updated at once.

    Args:
        learning_rate (float): Learning rate. Default ``1e-3``.
        beta1 (float): First-moment (rotated-basis) decay. Default ``0.9``.
        beta2 (float): Second-moment / escape decay. Default ``0.999``.
        eps (float): Stability constant added after the matrix sqrt.
            Default ``1e-6``.
        relative_epsilon (bool): Scale ``eps_root`` by the largest
            second-moment eigenvalue rather than using it as an absolute
            floor. Default ``False``.
        clip_unmagnified_grad (float): Global clipping threshold (across all
            pairs) for the unmagnified LoRA gradients. ``0`` disables.
            Default ``1.0``.
        update_capping (float): Per-update RMS cap applied to the first
            moment after preconditioning. ``0`` disables. Default ``0.0``.
        update_skipping (float): Zero out an unmagnified update whose RMS
            exceeds this threshold (outlier rejection). ``0`` disables.
            Default ``1.0``.
        weight_decay (float): Coupled weight-decay coefficient, added
            directly into the (rotated) update before scaling by ``-lr``.
            Default ``0.0``.
        apply_escape (bool): Apply the RITE escape correction when rotating
            second-moment matrices into a new QR basis. Default ``False``.
        lora_l_dim (int): Axis of the left (``A``) factor that holds the
            LoRA rank. Default ``0``.
        lora_r_dim (int): Axis of the right (``B``) factor that holds the
            LoRA rank. Default ``-1``.
        maybe_inf_to_nan (bool): Convert ±Inf to NaN before RMS threshold
            checks (so they don't silently pass clipping/skipping
            comparisons as enormous-but-finite values). Default ``True``.
        balance_param (bool): After applying the update, rescale each factor
            so ``‖A‖`` and ``‖B‖`` are equalised (geometric mean), which can
            improve conditioning of the ``A·B`` product. Default ``False``.
        maximize (bool): Maximise the objective instead of minimising.
            Default ``False``.
        name (str): Optimizer name.

    Example::

        # trainable_variables must alternate: [a1, b1, a2, b2, ...]
        optimizer = LoRARite(learning_rate=1e-3, apply_escape=True)
        optimizer.apply_gradients(zip(gradients, lora_variables))
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        relative_epsilon: bool = False,
        clip_unmagnified_grad: float = 1.0,
        update_capping: float = 0.0,
        update_skipping: float = 1.0,
        weight_decay: float = 0.0,
        apply_escape: bool = False,
        lora_l_dim: int = 0,
        lora_r_dim: int = -1,
        maybe_inf_to_nan: bool = True,
        balance_param: bool = False,
        maximize: bool = False,
        name: str = 'LoRARite',
        **kwargs,
    ):
        if eps < 0.0:
            raise ValueError(f'eps must be non-negative; got {eps}')
        if clip_unmagnified_grad < 0.0:
            raise ValueError('clip_unmagnified_grad must be non-negative')
        if update_capping < 0.0:
            raise ValueError('update_capping must be non-negative')
        if update_skipping < 0.0:
            raise ValueError('update_skipping must be non-negative')

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eps_root = eps ** 2
        self.relative_epsilon = relative_epsilon
        self.clip_unmagnified_grad = clip_unmagnified_grad
        self.update_capping = update_capping
        self.update_skipping = update_skipping
        self.apply_escape = apply_escape
        self.lora_l_dim = lora_l_dim
        self.lora_r_dim = lora_r_dim
        self.maybe_inf_to_nan = maybe_inf_to_nan
        self.balance_param = balance_param
        self.maximize = maximize

    @staticmethod
    def _compute_move_info(shape: List[int], dim: int) -> Tuple[List[int], List[int]]:
        rank = len(shape)
        if rank == 0:
            shape, rank, dim = [1], 1, 0
        elif dim < 0:
            dim = rank + dim
        perm = [i for i in range(rank) if i != dim] + [dim]
        moved_shape = [shape[p] for p in perm]
        return moved_shape, perm

    @staticmethod
    def _inverse_perm(perm: List[int]) -> List[int]:
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return inv

    def _to_2d(self, tensor: tf.Tensor, perm: List[int]) -> tf.Tensor:
        moved = tf.transpose(tensor, perm=perm)
        last_dim = moved.shape[-1]
        return tf.reshape(moved, [-1, last_dim])

    def _restore_shape(
        self, tensor_2d: tf.Tensor, moved_shape: List[int], perm: List[int]
    ) -> tf.Tensor:
        reshaped = tf.reshape(tensor_2d, moved_shape)
        return tf.transpose(reshaped, perm=self._inverse_perm(perm))

    def _inf_to_nan(self, tensor: tf.Tensor) -> tf.Tensor:
        if not self.maybe_inf_to_nan:
            return tensor
        nan_val = tf.constant(float('nan'), dtype=tensor.dtype)
        return tf.where(tf.math.is_inf(tensor), tf.fill(tf.shape(tensor), nan_val), tensor)

    def _clip_update(self, update: tf.Tensor, clip_threshold: float) -> tf.Tensor:
        update_rms = self._inf_to_nan(_reduce_rms(update))
        return update / tf.maximum(1.0, update_rms / clip_threshold)

    def _skip_update(self, update: tf.Tensor, skip_threshold: float) -> tf.Tensor:
        update_rms = self._inf_to_nan(_reduce_rms(update))
        return tf.where(update_rms > skip_threshold, tf.zeros_like(update), update)

    def build(self, var_list: list) -> None:
        if self.built:
            return
        super().build(var_list)

        if len(var_list) % 2 != 0:
            raise ValueError(
                f'LoRARite expects an even number of variables (alternating '
                f'A/B factor pairs); got {len(var_list)}.'
            )
        n_pairs = len(var_list) // 2

        self.moved_shape_l: List[List[int]] = []
        self.perm_l: List[List[int]] = []
        self.moved_shape_r: List[List[int]] = []
        self.perm_r: List[List[int]] = []

        self.v_l: List[tf.Variable] = []
        self.v_r: List[tf.Variable] = []
        self.m_l: List[tf.Variable] = []
        self.m_r: List[tf.Variable] = []
        self.basis_l: List[tf.Variable] = []
        self.basis_r: List[tf.Variable] = []
        self.escape_l: List[tf.Variable] = []
        self.escape_r: List[tf.Variable] = []
        self.pair_step: List[tf.Variable] = []

        for i in range(n_pairs):
            param_left = var_list[2 * i]
            param_right = var_list[2 * i + 1]

            shape_l = param_left.shape.as_list()
            shape_r = param_right.shape.as_list()
            moved_shape_l, perm_l = self._compute_move_info(shape_l, self.lora_l_dim)
            moved_shape_r, perm_r = self._compute_move_info(shape_r, self.lora_r_dim)
            self.moved_shape_l.append(moved_shape_l)
            self.perm_l.append(perm_l)
            self.moved_shape_r.append(moved_shape_r)
            self.perm_r.append(perm_r)

            rank_l = moved_shape_l[-1]
            rank_r = moved_shape_r[-1]
            rows_l = moved_shape_l[:-1]
            rows_r = moved_shape_r[:-1]
            flat_rows_l = 1
            for d in rows_l:
                flat_rows_l *= d
            flat_rows_r = 1
            for d in rows_r:
                flat_rows_r *= d

            def _make(shape, name):
                v = tf.Variable(tf.zeros(shape, dtype=tf.float32),
                                 trainable=False, name=name)
                self._track_variable(v)
                return v

            self.v_l.append(_make((rank_l, rank_l), 'v_l'))
            self.v_r.append(_make((rank_r, rank_r), 'v_r'))
            self.m_l.append(_make((flat_rows_l, rank_l), 'm_l'))
            self.m_r.append(_make((flat_rows_r, rank_r), 'm_r'))
            self.basis_l.append(_make((flat_rows_l, rank_l), 'basis_l'))
            self.basis_r.append(_make((flat_rows_r, rank_r), 'basis_r'))
            self.escape_l.append(_make((), 'escape_l'))
            self.escape_r.append(_make((), 'escape_r'))

            step_var = tf.Variable(0, dtype=tf.int32, trainable=False, name='pair_step')
            self._track_variable(step_var)
            self.pair_step.append(step_var)

    def _build_pair_info(
        self,
        pair_idx: int,
        param_left: tf.Variable,
        param_right: tf.Variable,
        grad_left: tf.Tensor,
        grad_right: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        perm_l = self.perm_l[pair_idx]
        perm_r = self.perm_r[pair_idx]

        param_left_2d = self._to_2d(tf.cast(param_left, tf.float32), perm_l)
        param_right_2d = self._to_2d(tf.cast(param_right, tf.float32), perm_r)

        basis_left, rotate_left = tf.linalg.qr(param_left_2d)
        basis_right, rotate_right = tf.linalg.qr(param_right_2d)
        rotate_inv_left = tf.linalg.pinv(rotate_left)
        rotate_inv_right = tf.linalg.pinv(rotate_right)

        projection_left = tf.matmul(basis_right, self.basis_r[pair_idx], transpose_a=True)
        projection_right = tf.matmul(basis_left, self.basis_l[pair_idx], transpose_a=True)

        g_l = tf.cast(grad_left, tf.float32)
        g_r = tf.cast(grad_right, tf.float32)
        if self.maximize:
            g_l, g_r = -g_l, -g_r
        g_l = self._to_2d(g_l, perm_l)
        g_r = self._to_2d(g_r, perm_r)

        update_left = tf.matmul(g_l, rotate_inv_right)
        update_right = tf.matmul(g_r, rotate_inv_left)

        if self.update_skipping > 0.0:
            update_left = self._skip_update(update_left, self.update_skipping)
            update_right = self._skip_update(update_right, self.update_skipping)

        return {
            'basis_l': basis_left, 'basis_r': basis_right,
            'rotate_inv_l': rotate_inv_left, 'rotate_inv_r': rotate_inv_right,
            'update_l': update_left, 'update_r': update_right,
            'projection_l': projection_left, 'projection_r': projection_right,
            'param_left_2d': param_left_2d, 'param_right_2d': param_right_2d,
            'param_left': param_left, 'param_right': param_right,
        }

    def _apply_pair_update(
        self, pair_idx: int, info: Dict[str, tf.Tensor], lr: tf.Tensor, grad_norm: tf.Tensor
    ) -> None:
        beta1, beta2 = self.beta1, self.beta2
        step = self.pair_step[pair_idx]

        update_left  = info['update_l']
        update_right = info['update_r']
        rotate_inv_left  = info['rotate_inv_l']
        rotate_inv_right = info['rotate_inv_r']
        projection_left  = info['projection_l']
        projection_right = info['projection_r']
        param_left_2d  = info['param_left_2d']
        param_right_2d = info['param_right_2d']

        if self.clip_unmagnified_grad > 0.0:
            scale = tf.where(
                grad_norm > self.clip_unmagnified_grad,
                self.clip_unmagnified_grad / tf.maximum(grad_norm, 1e-12),
                tf.constant(1.0, dtype=tf.float32),
            )
            update_left = update_left * scale
            update_right = update_right * scale

        second_left  = _compute_second_moment(update_left)
        second_right = _compute_second_moment(update_right)

        transformed_v_left  = _transform_second_moment(self.v_l[pair_idx], projection_left)
        transformed_v_right = _transform_second_moment(self.v_r[pair_idx], projection_right)

        if self.apply_escape:
            escape_left = _get_unmagnified_rotate_second_escape(
                transformed_v_left, self.v_l[pair_idx]
            )
            escape_right = _get_unmagnified_rotate_second_escape(
                transformed_v_right, self.v_r[pair_idx]
            )
            escape_left = _update_second_escape(
                step, tf.zeros_like(escape_left), escape_left + self.escape_l[pair_idx], beta2
            )
            escape_right = _update_second_escape(
                step, tf.zeros_like(escape_right), escape_right + self.escape_r[pair_idx], beta2
            )
        else:
            escape_left = tf.constant(0.0, dtype=tf.float32)
            escape_right = tf.constant(0.0, dtype=tf.float32)

        v_left  = _update_second_moment(step, second_left, transformed_v_left, beta2)
        v_right = _update_second_moment(step, second_right, transformed_v_right, beta2)

        update_left = _get_preconditioned_update(
            update_left, v_left, escape_left,
            self.eps, self.eps_root, self.relative_epsilon, self.apply_escape,
        )
        update_right = _get_preconditioned_update(
            update_right, v_right, escape_right,
            self.eps, self.eps_root, self.relative_epsilon, self.apply_escape,
        )

        m_left  = _transform_first_moment(self.m_l[pair_idx], projection_left)
        m_right = _transform_first_moment(self.m_r[pair_idx], projection_right)
        m_left  = _update_first_moment(step, update_left, m_left, beta1)
        m_right = _update_first_moment(step, update_right, m_right, beta1)

        if self.update_capping > 0.0:
            m_left  = self._clip_update(m_left, self.update_capping)
            m_right = self._clip_update(m_right, self.update_capping)

        update_left  = tf.matmul(m_left,  rotate_inv_right, transpose_b=True)
        update_right = tf.matmul(m_right, rotate_inv_left,  transpose_b=True)

        if self.weight_decay is not None and self.weight_decay > 0.0:
            update_left  = update_left  + param_left_2d  * self.weight_decay
            update_right = update_right + param_right_2d * self.weight_decay

        update_left  = update_left  * (-lr)
        update_right = update_right * (-lr)

        if self.balance_param:
            left_norm  = tf.norm(param_left_2d  + update_left)  + 1e-6
            right_norm = tf.norm(param_right_2d + update_right) + 1e-6
            balanced_norm = tf.sqrt(left_norm * right_norm)
            update_left = (
                update_left * (balanced_norm / left_norm)
                + param_left_2d * (balanced_norm / left_norm - 1.0)
            )
            update_right = (
                update_right * (balanced_norm / right_norm)
                + param_right_2d * (balanced_norm / right_norm - 1.0)
            )

        moved_shape_l = self.moved_shape_l[pair_idx]
        moved_shape_r = self.moved_shape_r[pair_idx]
        perm_l = self.perm_l[pair_idx]
        perm_r = self.perm_r[pair_idx]

        delta_left  = self._restore_shape(update_left,  moved_shape_l, perm_l)
        delta_right = self._restore_shape(update_right, moved_shape_r, perm_r)

        param_left  = info['param_left']
        param_right = info['param_right']
        param_left.assign_add(tf.cast(delta_left, param_left.dtype))
        param_right.assign_add(tf.cast(delta_right, param_right.dtype))

        self.pair_step[pair_idx].assign_add(1)
        self.v_l[pair_idx].assign(v_left)
        self.v_r[pair_idx].assign(v_right)
        self.m_l[pair_idx].assign(m_left)
        self.m_r[pair_idx].assign(m_right)
        self.basis_l[pair_idx].assign(info['basis_l'])
        self.basis_r[pair_idx].assign(info['basis_r'])
        self.escape_l[pair_idx].assign(escape_left)
        self.escape_r[pair_idx].assign(escape_right)

    def update_step(self, grads: list, variables: list, learning_rate) -> None:
        lr = tf.cast(learning_rate, tf.float32)

        pair_sides: Dict[int, Dict[str, Tuple[tf.Tensor, tf.Variable]]] = {}
        for g, v in zip(grads, variables):
            global_idx = self._get_variable_index(v)
            pair_idx = global_idx // 2
            side = 'left' if global_idx % 2 == 0 else 'right'
            pair_sides.setdefault(pair_idx, {})[side] = (g, v)

        pair_infos: Dict[int, Dict[str, tf.Tensor]] = {}
        grad_norm_sq = tf.constant(0.0, dtype=tf.float32)

        for pair_idx, sides in pair_sides.items():
            if 'left' not in sides or 'right' not in sides:
                continue
            g_l, p_l = sides['left']
            g_r, p_r = sides['right']

            info = self._build_pair_info(pair_idx, p_l, p_r, g_l, g_r)
            grad_norm_sq += tf.reduce_sum(tf.square(info['update_l']))
            grad_norm_sq += tf.reduce_sum(tf.square(info['update_r']))
            pair_infos[pair_idx] = info

        if not pair_infos:
            return

        grad_norm = tf.sqrt(grad_norm_sq)

        for pair_idx, info in pair_infos.items():
            self._apply_pair_update(pair_idx, info, lr, grad_norm)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'beta1':                  self.beta1,
            'beta2':                  self.beta2,
            'eps':                    self.eps,
            'relative_epsilon':       self.relative_epsilon,
            'clip_unmagnified_grad':  self.clip_unmagnified_grad,
            'update_capping':         self.update_capping,
            'update_skipping':        self.update_skipping,
            'apply_escape':           self.apply_escape,
            'lora_l_dim':             self.lora_l_dim,
            'lora_r_dim':             self.lora_r_dim,
            'maybe_inf_to_nan':       self.maybe_inf_to_nan,
            'balance_param':          self.balance_param,
            'maximize':               self.maximize,
        })
        return config


class LoRARite_e(optimizer.Optimizer):
    """Enhanced LoRARite with the full BaseOptimizer technique set.

    .. important::
        Same pairing requirement as :class:`LoRARite`: variables must
        alternate ``lora_a_1, lora_b_1, lora_a_2, lora_b_2, ...``.

    Args (in addition to the base LoRARite set):
        orthograd (bool): Orthogonalize each factor's raw gradient against
            its own weight before anything else touches it. Handled
            automatically by ``BaseOptimizer``. Default ``False``.
        gc (bool): Gradient Centralization on each factor's raw
            gradient (original shape) before rotation. Default ``False``.
        agc (bool): Adaptive Gradient Clipping on each factor's raw
            gradient, unit-wise against that factor's own weight norm.
            Default ``False``.
        agc_clip_val (float): AGC clipping ratio. Default ``1e-2``.
        agc_eps (float): AGC minimum weight norm. Default ``1e-3``.
        cautious (bool): Mask the rotated first-moment update to components
            that agree in sign with the unmagnified gradient, in the
            rotated 2D basis (Cautious-Adam style). Default ``False``.
        trust_ratio (bool): LARS-style layer-wise step-size scaling of the
            final delta, computed in the *original* parameter space
            (``‖param‖ / ‖delta‖``). Default ``False``.
        trust_clip (bool): Clip the trust ratio to ≤ 1. Default ``False``.
        pnm (bool): Smooth the final per-factor delta (original parameter
            space) with Positive-Negative Momentum, *in addition to*
            LoRARite's own rotated-basis first moment — not a replacement.
            Default ``False``.
        lookahead (bool): Lookahead slow-weight blending of the actual
            factor variables. Default ``False``.
        lookahead_merge_time (int): Lookahead sync frequency (in optimizer
            steps). Default ``5``.
        lookahead_blending_alpha (float): Lookahead blend factor. Default ``0.5``.
        weight_decouple (bool): Apply weight decay multiplicatively,
            directly to the parameter, *before* the QR decomposition
            (AdamW-style decoupled decay) instead of additively folding it
            into the rotated update. Default ``False`` (matches original
            LoRARite's coupled behaviour).
        fixed_decay (bool): When ``weight_decouple=True``, apply
            ``param *= 1 - weight_decay`` without scaling by ``lr``.
            Default ``False``.

    Example::

        optimizer = LoRARite_e(
            learning_rate=1e-3,
            apply_escape=True,
            cautious=True,
            trust_ratio=True,
            weight_decouple=True,
            weight_decay=1e-2,
        )
        optimizer.apply_gradients(zip(gradients, lora_variables))
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        relative_epsilon: bool = False,
        clip_unmagnified_grad: float = 1.0,
        update_capping: float = 0.0,
        update_skipping: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        apply_escape: bool = False,
        lora_l_dim: int = 0,
        lora_r_dim: int = -1,
        maybe_inf_to_nan: bool = True,
        balance_param: bool = False,
        maximize: bool = False,
        orthograd: bool = False,
        gc: bool = False,
        agc: bool = False,
        agc_clip_val: float = 1e-2,
        agc_eps: float = 1e-3,
        cautious: bool = False,
        trust_ratio: bool = False,
        trust_clip: bool = False,
        pnm: bool = False,
        lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        name: str = 'LoRARite_e',
        **kwargs,
    ):
        if eps < 0.0:
            raise ValueError(f'eps must be non-negative; got {eps}')
        if clip_unmagnified_grad < 0.0:
            raise ValueError('clip_unmagnified_grad must be non-negative')
        if update_capping < 0.0:
            raise ValueError('update_capping must be non-negative')
        if update_skipping < 0.0:
            raise ValueError('update_skipping must be non-negative')

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eps_root = eps ** 2
        self.relative_epsilon = relative_epsilon
        self.clip_unmagnified_grad = clip_unmagnified_grad
        self.update_capping = update_capping
        self.update_skipping = update_skipping
        self.apply_escape = apply_escape
        self.lora_l_dim = lora_l_dim
        self.lora_r_dim = lora_r_dim
        self.maybe_inf_to_nan = maybe_inf_to_nan
        self.balance_param = balance_param
        self.maximize = maximize

        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay

        self.orthograd = orthograd

        self.gc = gc
        self.agc = agc
        self.agc_clip_val = agc_clip_val
        self.agc_eps = agc_eps

        self.cautious = cautious

        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.pnm = pnm
        self.lookahead = lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha

    @staticmethod
    def _compute_move_info(shape: List[int], dim: int) -> Tuple[List[int], List[int]]:
        rank = len(shape)
        if rank == 0:
            shape, rank, dim = [1], 1, 0
        elif dim < 0:
            dim = rank + dim
        perm = [i for i in range(rank) if i != dim] + [dim]
        moved_shape = [shape[p] for p in perm]
        return moved_shape, perm

    @staticmethod
    def _inverse_perm(perm: List[int]) -> List[int]:
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return inv

    def _to_2d(self, tensor: tf.Tensor, perm: List[int]) -> tf.Tensor:
        moved = tf.transpose(tensor, perm=perm)
        last_dim = moved.shape[-1]
        return tf.reshape(moved, [-1, last_dim])

    def _restore_shape(
        self, tensor_2d: tf.Tensor, moved_shape: List[int], perm: List[int]
    ) -> tf.Tensor:
        reshaped = tf.reshape(tensor_2d, moved_shape)
        return tf.transpose(reshaped, perm=self._inverse_perm(perm))

    def _inf_to_nan(self, tensor: tf.Tensor) -> tf.Tensor:
        if not self.maybe_inf_to_nan:
            return tensor
        nan_val = tf.constant(float('nan'), dtype=tensor.dtype)
        return tf.where(tf.math.is_inf(tensor), tf.fill(tf.shape(tensor), nan_val), tensor)

    def _clip_update(self, update: tf.Tensor, clip_threshold: float) -> tf.Tensor:
        update_rms = self._inf_to_nan(_reduce_rms(update))
        return update / tf.maximum(1.0, update_rms / clip_threshold)

    def _skip_update(self, update: tf.Tensor, skip_threshold: float) -> tf.Tensor:
        update_rms = self._inf_to_nan(_reduce_rms(update))
        return tf.where(update_rms > skip_threshold, tf.zeros_like(update), update)

    def _apply_decoupled_decay(self, variable: tf.Variable, lr: tf.Tensor) -> None:
        wd = tf.cast(self.weight_decay, variable.dtype)
        if self.fixed_decay:
            factor = 1.0 - wd
        else:
            factor = 1.0 - wd * tf.cast(lr, variable.dtype)
        variable.assign(variable * factor)

    def build(self, var_list: list) -> None:
        if self.built:
            return
        super().build(var_list)

        if len(var_list) % 2 != 0:
            raise ValueError(
                f'LoRARite_e expects an even number of variables (alternating '
                f'A/B factor pairs); got {len(var_list)}.'
            )
        n_pairs = len(var_list) // 2

        self.moved_shape_l: List[List[int]] = []
        self.perm_l: List[List[int]] = []
        self.moved_shape_r: List[List[int]] = []
        self.perm_r: List[List[int]] = []

        self.v_l: List[tf.Variable] = []
        self.v_r: List[tf.Variable] = []
        self.m_l: List[tf.Variable] = []
        self.m_r: List[tf.Variable] = []
        self.basis_l: List[tf.Variable] = []
        self.basis_r: List[tf.Variable] = []
        self.escape_l: List[tf.Variable] = []
        self.escape_r: List[tf.Variable] = []
        self.pair_step: List[tf.Variable] = []

        for i in range(n_pairs):
            param_left = var_list[2 * i]
            param_right = var_list[2 * i + 1]

            shape_l = param_left.shape.as_list()
            shape_r = param_right.shape.as_list()
            moved_shape_l, perm_l = self._compute_move_info(shape_l, self.lora_l_dim)
            moved_shape_r, perm_r = self._compute_move_info(shape_r, self.lora_r_dim)
            self.moved_shape_l.append(moved_shape_l)
            self.perm_l.append(perm_l)
            self.moved_shape_r.append(moved_shape_r)
            self.perm_r.append(perm_r)

            rank_l = moved_shape_l[-1]
            rank_r = moved_shape_r[-1]
            flat_rows_l = 1
            for d in moved_shape_l[:-1]:
                flat_rows_l *= d
            flat_rows_r = 1
            for d in moved_shape_r[:-1]:
                flat_rows_r *= d

            def _make(shape, name):
                v = tf.Variable(tf.zeros(shape, dtype=tf.float32),
                                 trainable=False, name=name)
                self._track_variable(v)
                return v

            self.v_l.append(_make((rank_l, rank_l), 'v_l'))
            self.v_r.append(_make((rank_r, rank_r), 'v_r'))
            self.m_l.append(_make((flat_rows_l, rank_l), 'm_l'))
            self.m_r.append(_make((flat_rows_r, rank_r), 'm_r'))
            self.basis_l.append(_make((flat_rows_l, rank_l), 'basis_l'))
            self.basis_r.append(_make((flat_rows_r, rank_r), 'basis_r'))
            self.escape_l.append(_make((), 'escape_l'))
            self.escape_r.append(_make((), 'escape_r'))

            step_var = tf.Variable(0, dtype=tf.int32, trainable=False, name='pair_step')
            self._track_variable(step_var)
            self.pair_step.append(step_var)

    def _build_pair_info(
        self,
        pair_idx: int,
        param_left: tf.Variable,
        param_right: tf.Variable,
        grad_left: tf.Tensor,
        grad_right: tf.Tensor,
        lr: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        perm_l = self.perm_l[pair_idx]
        perm_r = self.perm_r[pair_idx]

        if self.weight_decouple and self.weight_decay is not None and self.weight_decay > 0.0:
            self._apply_decoupled_decay(param_left, lr)
            self._apply_decoupled_decay(param_right, lr)

        param_left_2d = self._to_2d(tf.cast(param_left, tf.float32), perm_l)
        param_right_2d = self._to_2d(tf.cast(param_right, tf.float32), perm_r)

        basis_left, rotate_left = tf.linalg.qr(param_left_2d)
        basis_right, rotate_right = tf.linalg.qr(param_right_2d)
        rotate_inv_left = tf.linalg.pinv(rotate_left)
        rotate_inv_right = tf.linalg.pinv(rotate_right)

        projection_left = tf.matmul(basis_right, self.basis_r[pair_idx], transpose_a=True)
        projection_right = tf.matmul(basis_left, self.basis_l[pair_idx], transpose_a=True)

        g_l = tf.cast(grad_left, tf.float32)
        g_r = tf.cast(grad_right, tf.float32)
        if self.maximize:
            g_l, g_r = -g_l, -g_r

        if self.gc:
            g_l = self.apply_gc(g_l)
            g_r = self.apply_gc(g_r)
        if self.agc:
            g_l = self.apply_agc(param_left, g_l, agc_eps=self.agc_eps, agc_clip_val=self.agc_clip_val)
            g_r = self.apply_agc(param_right, g_r, agc_eps=self.agc_eps, agc_clip_val=self.agc_clip_val)

        g_l = self._to_2d(g_l, perm_l)
        g_r = self._to_2d(g_r, perm_r)

        update_left = tf.matmul(g_l, rotate_inv_right)
        update_right = tf.matmul(g_r, rotate_inv_left)

        if self.update_skipping > 0.0:
            update_left = self._skip_update(update_left, self.update_skipping)
            update_right = self._skip_update(update_right, self.update_skipping)

        return {
            'basis_l': basis_left, 'basis_r': basis_right,
            'rotate_inv_l': rotate_inv_left, 'rotate_inv_r': rotate_inv_right,
            'update_l': update_left, 'update_r': update_right,
            'projection_l': projection_left, 'projection_r': projection_right,
            'param_left_2d': param_left_2d, 'param_right_2d': param_right_2d,
            'param_left': param_left, 'param_right': param_right,
        }

    def _apply_pair_update(
        self,
        pair_idx: int,
        info: Dict[str, tf.Tensor],
        lr: tf.Tensor,
        grad_norm: tf.Tensor,
        global_step: tf.Tensor,
    ) -> None:
        beta1, beta2 = self.beta1, self.beta2
        step = self.pair_step[pair_idx]

        update_left  = info['update_l']
        update_right = info['update_r']
        rotate_inv_left  = info['rotate_inv_l']
        rotate_inv_right = info['rotate_inv_r']
        projection_left  = info['projection_l']
        projection_right = info['projection_r']
        param_left_2d  = info['param_left_2d']
        param_right_2d = info['param_right_2d']
        param_left  = info['param_left']
        param_right = info['param_right']

        if self.clip_unmagnified_grad > 0.0:
            scale = tf.where(
                grad_norm > self.clip_unmagnified_grad,
                self.clip_unmagnified_grad / tf.maximum(grad_norm, 1e-12),
                tf.constant(1.0, dtype=tf.float32),
            )
            update_left = update_left * scale
            update_right = update_right * scale

        second_left  = _compute_second_moment(update_left)
        second_right = _compute_second_moment(update_right)

        transformed_v_left  = _transform_second_moment(self.v_l[pair_idx], projection_left)
        transformed_v_right = _transform_second_moment(self.v_r[pair_idx], projection_right)

        if self.apply_escape:
            escape_left = _get_unmagnified_rotate_second_escape(
                transformed_v_left, self.v_l[pair_idx]
            )
            escape_right = _get_unmagnified_rotate_second_escape(
                transformed_v_right, self.v_r[pair_idx]
            )
            escape_left = _update_second_escape(
                step, tf.zeros_like(escape_left), escape_left + self.escape_l[pair_idx], beta2
            )
            escape_right = _update_second_escape(
                step, tf.zeros_like(escape_right), escape_right + self.escape_r[pair_idx], beta2
            )
        else:
            escape_left = tf.constant(0.0, dtype=tf.float32)
            escape_right = tf.constant(0.0, dtype=tf.float32)

        v_left  = _update_second_moment(step, second_left, transformed_v_left, beta2)
        v_right = _update_second_moment(step, second_right, transformed_v_right, beta2)

        update_left = _get_preconditioned_update(
            update_left, v_left, escape_left,
            self.eps, self.eps_root, self.relative_epsilon, self.apply_escape,
        )
        update_right = _get_preconditioned_update(
            update_right, v_right, escape_right,
            self.eps, self.eps_root, self.relative_epsilon, self.apply_escape,
        )

        m_left  = _transform_first_moment(self.m_l[pair_idx], projection_left)
        m_right = _transform_first_moment(self.m_r[pair_idx], projection_right)
        m_left  = _update_first_moment(step, update_left, m_left, beta1)
        m_right = _update_first_moment(step, update_right, m_right, beta1)

        if self.update_capping > 0.0:
            m_left  = self._clip_update(m_left, self.update_capping)
            m_right = self._clip_update(m_right, self.update_capping)

        if self.cautious:
            m_left  = self.apply_cautious(m_left, info['update_l'])
            m_right = self.apply_cautious(m_right, info['update_r'])

        update_left  = tf.matmul(m_left,  rotate_inv_right, transpose_b=True)
        update_right = tf.matmul(m_right, rotate_inv_left,  transpose_b=True)

        if (
            not self.weight_decouple
            and self.weight_decay is not None
            and self.weight_decay > 0.0
        ):
            update_left  = update_left  + param_left_2d  * self.weight_decay
            update_right = update_right + param_right_2d * self.weight_decay

        update_left  = update_left  * (-lr)
        update_right = update_right * (-lr)

        if self.balance_param:
            left_norm  = tf.norm(param_left_2d  + update_left)  + 1e-6
            right_norm = tf.norm(param_right_2d + update_right) + 1e-6
            balanced_norm = tf.sqrt(left_norm * right_norm)
            update_left = (
                update_left * (balanced_norm / left_norm)
                + param_left_2d * (balanced_norm / left_norm - 1.0)
            )
            update_right = (
                update_right * (balanced_norm / right_norm)
                + param_right_2d * (balanced_norm / right_norm - 1.0)
            )

        moved_shape_l = self.moved_shape_l[pair_idx]
        moved_shape_r = self.moved_shape_r[pair_idx]
        perm_l = self.perm_l[pair_idx]
        perm_r = self.perm_r[pair_idx]

        delta_left  = self._restore_shape(update_left,  moved_shape_l, perm_l)
        delta_right = self._restore_shape(update_right, moved_shape_r, perm_r)

        if self.trust_ratio:
            delta_left  = self.apply_trust_ratio(param_left,  delta_left)
            delta_right = self.apply_trust_ratio(param_right, delta_right)

        if self.pnm:
            idx_left  = self._get_variable_index(param_left)
            idx_right = self._get_variable_index(param_right)
            delta_left  = self.apply_pnm(delta_left,  step, idx_left)
            delta_right = self.apply_pnm(delta_right, step, idx_right)

        param_left.assign_add(tf.cast(delta_left, param_left.dtype))
        param_right.assign_add(tf.cast(delta_right, param_right.dtype))

        if self.lookahead:
            self.lookahead_merge(param_left, global_step)
            self.lookahead_merge(param_right, global_step)

        self.pair_step[pair_idx].assign_add(1)
        self.v_l[pair_idx].assign(v_left)
        self.v_r[pair_idx].assign(v_right)
        self.m_l[pair_idx].assign(m_left)
        self.m_r[pair_idx].assign(m_right)
        self.basis_l[pair_idx].assign(info['basis_l'])
        self.basis_r[pair_idx].assign(info['basis_r'])
        self.escape_l[pair_idx].assign(escape_left)
        self.escape_r[pair_idx].assign(escape_right)

    def update_step(self, grads: list, variables: list, learning_rate) -> None:
        lr = tf.cast(learning_rate, tf.float32)
        global_step = self._iterations + 1

        pair_sides: Dict[int, Dict[str, Tuple[tf.Tensor, tf.Variable]]] = {}
        for g, v in zip(grads, variables):
            global_idx = self._get_variable_index(v)
            pair_idx = global_idx // 2
            side = 'left' if global_idx % 2 == 0 else 'right'
            pair_sides.setdefault(pair_idx, {})[side] = (g, v)

        pair_infos: Dict[int, Dict[str, tf.Tensor]] = {}
        grad_norm_sq = tf.constant(0.0, dtype=tf.float32)

        for pair_idx, sides in pair_sides.items():
            if 'left' not in sides or 'right' not in sides:
                continue
            g_l, p_l = sides['left']
            g_r, p_r = sides['right']

            info = self._build_pair_info(pair_idx, p_l, p_r, g_l, g_r, lr)
            grad_norm_sq += tf.reduce_sum(tf.square(info['update_l']))
            grad_norm_sq += tf.reduce_sum(tf.square(info['update_r']))
            pair_infos[pair_idx] = info

        if not pair_infos:
            return

        grad_norm = tf.sqrt(grad_norm_sq)

        for pair_idx, info in pair_infos.items():
            self._apply_pair_update(pair_idx, info, lr, grad_norm, global_step)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'beta1':                    self.beta1,
            'beta2':                    self.beta2,
            'eps':                      self.eps,
            'relative_epsilon':         self.relative_epsilon,
            'clip_unmagnified_grad':    self.clip_unmagnified_grad,
            'update_capping':           self.update_capping,
            'update_skipping':          self.update_skipping,
            'weight_decouple':          self.weight_decouple,
            'fixed_decay':              self.fixed_decay,
            'apply_escape':             self.apply_escape,
            'lora_l_dim':               self.lora_l_dim,
            'lora_r_dim':               self.lora_r_dim,
            'maybe_inf_to_nan':         self.maybe_inf_to_nan,
            'balance_param':            self.balance_param,
            'maximize':                 self.maximize,
            'orthograd':                self.orthograd,
            'gc':                   self.gc,
            'agc':                      self.agc,
            'agc_clip_val':             self.agc_clip_val,
            'agc_eps':                  self.agc_eps,
            'cautious':                 self.cautious,
            'trust_ratio':              self.trust_ratio,
            'trust_clip':               self.trust_clip,
            'pnm':                      self.pnm,
            'lookahead':                self.lookahead,
            'lookahead_merge_time':     self.lookahead_merge_time,
            'lookahead_blending_alpha': self.lookahead_blending_alpha,
        })
        return config