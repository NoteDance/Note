import tensorflow as tf
from Note.nn.optimizer import optimizer
import math
from typing import Dict, List, Optional, Tuple
import warnings


GROUP_SIZE: int = 32


def quantize_state(
    tensor: tf.Tensor,
    signed: bool = True,
    sqrt: bool = False,
    softsign: bool = True,
    group_size: int = GROUP_SIZE,
) -> Tuple[tf.Tensor, tf.Tensor]:
    original_shape = tf.shape(tensor)
    numel = tf.size(tensor)
    values = tf.cast(tf.reshape(tensor, [-1]), tf.float32)

    if sqrt:
        values = tf.sqrt(tf.maximum(values, 0.0))

    pad = tf.math.floormod(-numel, group_size)
    values_padded = tf.concat(
        [values, tf.zeros([pad], dtype=tf.float32)], axis=0
    )

    groups = tf.reshape(values_padded, [-1, group_size])
    scales = tf.reduce_max(tf.abs(groups), axis=1)
    scales = tf.maximum(scales, 1e-12)
    normalized = groups / tf.expand_dims(scales, 1)

    if softsign:
        normalized = 2.0 * normalized / (1.0 + tf.abs(normalized))

    quant_max = 127.0 if signed else 255.0
    quant_min = -127.0 if signed else 0.0
    quantized = tf.clip_by_value(
        tf.round(normalized * quant_max), quant_min, quant_max
    )
    quantized = tf.reshape(tf.reshape(quantized, [-1])[:numel], original_shape)

    target_dtype = tf.int8 if signed else tf.uint8
    return tf.cast(quantized, target_dtype), tf.cast(scales, tf.float16)


def dequantize_state(
    quantized: tf.Tensor,
    scales: tf.Tensor,
    signed: bool = True,
    sqrt: bool = False,
    softsign: bool = True,
    group_size: int = GROUP_SIZE,
) -> tf.Tensor:
    original_shape = tf.shape(quantized)
    numel = tf.size(quantized)
    values = tf.cast(tf.reshape(quantized, [-1]), tf.float32)

    pad = tf.math.floormod(-numel, group_size)
    values_padded = tf.concat(
        [values, tf.zeros([pad], dtype=tf.float32)], axis=0
    )

    quant_max = 127.0 if signed else 255.0
    groups = tf.reshape(values_padded, [-1, group_size]) / quant_max

    if softsign:
        groups = groups / tf.maximum(2.0 - tf.abs(groups), 1e-12)

    restored = groups * tf.reshape(tf.cast(scales, tf.float32), [-1, 1])
    restored = tf.reshape(tf.reshape(restored, [-1])[:numel], original_shape)
    return tf.square(restored) if sqrt else restored


_STATE_SPECS: dict = {
    'exp_avg':    (True,  False, True),
    'exp_avg_sq': (False, True,  False),
}


_TF_DTYPE_BYTES: dict = {
    tf.float16:  2,
    tf.bfloat16: 2,
    tf.float32:  4,
    tf.float64:  8,
}


def ulp_scale(narrow: tf.Tensor) -> tf.Tensor:
    abs_vals = tf.abs(tf.cast(narrow, tf.float32))
    eps_half = (
        tf.constant(2.0 ** -11, dtype=tf.float32)
        if narrow.dtype == tf.float16
        else tf.constant(2.0 ** -8, dtype=tf.float32)
    )
    return tf.maximum(abs_vals * eps_half, tf.constant(1.1754944e-38, dtype=tf.float32))


def compute_ecc_bits(
    fp32_param: tf.Tensor,
    narrow_param: tf.Tensor,
    master_byte_width: int,
) -> tf.Tensor:
    error_bytes = master_byte_width - _TF_DTYPE_BYTES[narrow_param.dtype]
    if error_bytes == 1:
        signed_max, error_dtype = 127.0, tf.int8
    elif error_bytes == 2:
        signed_max, error_dtype = 32767.0, tf.int16
    else:
        raise ValueError(
            f'master_byte_width={master_byte_width} gives unsupported '
            f'error_bytes={error_bytes} for dtype {narrow_param.dtype}'
        )
    normalized = (
        (fp32_param - tf.cast(narrow_param, tf.float32))
        / ulp_scale(narrow_param)
    )
    return tf.cast(
        tf.round(tf.clip_by_value(normalized, -1.0, 1.0) * signed_max),
        error_dtype,
    )


def reconstruct_fp32_param(
    narrow_param: tf.Tensor, error_bits: tf.Tensor
) -> tf.Tensor:
    """Reconstruct the fp32 master weight from the narrow copy + ECC bits."""
    signed_max = 127.0 if error_bits.dtype == tf.int8 else 32767.0
    correction = (
        tf.cast(error_bits, tf.float32) / signed_max * ulp_scale(narrow_param)
    )
    return tf.cast(narrow_param, tf.float32) + correction


class FlashAdamW(optimizer.Optimizer):
    """FlashAdamW with compressed optimizer states for TensorFlow.

    Args:
        learning_rate (float): Learning rate. Default: ``1e-3``.
        beta1 (float): Exponential decay for the first moment. Default: ``0.9``.
        beta2 (float): Exponential decay for the second moment. Default: ``0.999``.
        eps (float): Numerical stability constant. Default: ``1e-8``.
        weight_decay (float): Decoupled weight-decay coefficient. Default: ``1e-2``.
        quantize (bool): Store Adam moments as grouped 8-bit values plus fp16
            scale factors.  Default: ``True``.
        master_weight_bits (int | None): Effective master-weight precision for
            bf16/fp16 parameters.  Supported values: ``None`` (disabled),
            ``24`` (bf16 param + int8 ECC ≈ 24-bit), ``32`` (fp16/bf16 + int16
            ECC ≈ 32-bit).  Default: ``None``.
        maximize (bool): Maximise the objective instead of minimising.
            Default: ``False``.
        name (str): Optimizer name.

    Example::

        optimizer = FlashAdamW(
            learning_rate=1e-3,
            quantize=True,
            master_weight_bits=None,
        )
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    """

    _BITS_TO_BYTES: dict = {None: 0, 24: 3, 32: 4}
    _VALID_MASTER_WEIGHT_BITS: tuple = (None, 24, 32)

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        quantize: bool = True,
        master_weight_bits: Optional[int] = None,
        maximize: bool = False,
        name: str = 'FlashAdamW',
        **kwargs,
    ):
        if master_weight_bits not in self._VALID_MASTER_WEIGHT_BITS:
            raise ValueError(
                f'master_weight_bits must be one of {self._VALID_MASTER_WEIGHT_BITS}; '
                f'got {master_weight_bits!r}'
            )

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.quantize = quantize
        self._master_weight_bits = master_weight_bits       # kept for get_config
        self.master_byte_width = self._BITS_TO_BYTES[master_weight_bits]
        self.maximize = maximize

        self.initial_lr: float = float(learning_rate)

        self.orthograd = False

    def build(self, var_list: list) -> None:
        if self.built:
            return
        super().build(var_list)

        self.exp_avg: List = []

        self.q_exp_avg:    List[tf.Variable] = []   # int8
        self.s_exp_avg:    List[tf.Variable] = []   # fp16 scales
        self.q_exp_avg_sq: List[tf.Variable] = []   # uint8
        self.s_exp_avg_sq: List[tf.Variable] = []   # fp16 scales

        self.error_bits: List[Optional[tf.Variable]] = []

        for var in var_list:
            numel: int = var.shape.num_elements() or int(tf.size(var).numpy())
            num_groups: int = max(math.ceil(numel / GROUP_SIZE), 1)

            if self.quantize:
                q_m1 = tf.Variable(
                    tf.zeros(var.shape, dtype=tf.int8),
                    trainable=False, name='q_exp_avg',
                )
                s_m1 = tf.Variable(
                    tf.zeros([num_groups], dtype=tf.float16),
                    trainable=False, name='s_exp_avg',
                )
                self.q_exp_avg.append(q_m1)
                self.s_exp_avg.append(s_m1)
                self._track_variable(q_m1)
                self._track_variable(s_m1)

                q_m2 = tf.Variable(
                    tf.zeros(var.shape, dtype=tf.uint8),
                    trainable=False, name='q_exp_avg_sq',
                )
                s_m2 = tf.Variable(
                    tf.zeros([num_groups], dtype=tf.float16),
                    trainable=False, name='s_exp_avg_sq',
                )
                self.q_exp_avg_sq.append(q_m2)
                self.s_exp_avg_sq.append(s_m2)
                self._track_variable(q_m2)
                self._track_variable(s_m2)
            else:
                self.exp_avg.append(
                    self.add_variable_from_reference(var, name='exp_avg')
                )
                self.exp_avg_sq.append(
                    self.add_variable_from_reference(var, name='exp_avg_sq')
                )

            var_bytes = _TF_DTYPE_BYTES.get(var.dtype, 0)
            error_bytes = (
                self.master_byte_width - var_bytes
                if (var_bytes and var.dtype in (tf.float16, tf.bfloat16))
                else 0
            )
            if error_bytes > 0:
                error_dtype = tf.int8 if error_bytes == 1 else tf.int16
                ecc_var = tf.Variable(
                    tf.zeros(var.shape, dtype=error_dtype),
                    trainable=False, name='error_bits',
                )
                self.error_bits.append(ecc_var)
                self._track_variable(ecc_var)
            else:
                self.error_bits.append(None)

    def _materialize(self, name: str, idx: int) -> tf.Tensor:
        """Return the named optimizer state as a float32 tensor."""
        signed, sqrt, softsign = _STATE_SPECS[name]
        if self.quantize:
            q, s = (
                (self.q_exp_avg[idx],    self.s_exp_avg[idx])
                if name == 'exp_avg'
                else (self.q_exp_avg_sq[idx], self.s_exp_avg_sq[idx])
            )
            return dequantize_state(q, s, signed=signed, sqrt=sqrt, softsign=softsign)
        else:
            raw = self.exp_avg[idx] if name == 'exp_avg' else self.exp_avg_sq[idx]
            return tf.cast(raw, tf.float32)

    def _store(
        self,
        name: str,
        idx: int,
        value: tf.Tensor,
        var_dtype: tf.DType,
    ) -> None:
        signed, sqrt, softsign = _STATE_SPECS[name]
        if self.quantize:
            q, s = quantize_state(value, signed=signed, sqrt=sqrt, softsign=softsign)
            if name == 'exp_avg':
                self.q_exp_avg[idx].assign(q)
                self.s_exp_avg[idx].assign(s)
            else:
                self.q_exp_avg_sq[idx].assign(q)
                self.s_exp_avg_sq[idx].assign(s)
        else:
            target = self.exp_avg[idx] if name == 'exp_avg' else self.exp_avg_sq[idx]
            target.assign(tf.cast(value, var_dtype))

    def _get_param_fp32(self, var: tf.Variable, idx: int) -> tf.Tensor:
        ecc = self.error_bits[idx]
        if ecc is not None:
            return reconstruct_fp32_param(var, ecc)
        return tf.cast(var, tf.float32)

    def _set_param_fp32(
        self, var: tf.Variable, idx: int, value: tf.Tensor
    ) -> None:
        var.assign(tf.cast(value, var.dtype))
        ecc = self.error_bits[idx]
        if ecc is not None:
            ecc.assign(compute_ecc_bits(value, var, self.master_byte_width))

    def update_step(
        self,
        grads: list,
        variables: list,
        learning_rate,
    ) -> None:
        lr = tf.cast(learning_rate, tf.float32)
        step = tf.cast(self._iterations + 1, tf.float32)

        beta1_t = tf.constant(self.beta1, dtype=tf.float32)
        beta2_t = tf.constant(self.beta2, dtype=tf.float32)
        bias_corr1 = 1.0 - beta1_t ** step
        bias_corr2 = 1.0 - beta2_t ** step

        for gradient, variable in zip(grads, variables):
            if gradient is None:
                continue

            idx = self._get_variable_index(variable)
            g = tf.cast(gradient, tf.float32)

            if self.maximize:
                g = -g

            exp_avg    = self._materialize('exp_avg',    idx)
            exp_avg_sq = self._materialize('exp_avg_sq', idx)

            p = self._get_param_fp32(variable, idx)

            gradient = self.apply_weight_decay(variable, gradient, lr, 1.0 / self.initial_lr)

            exp_avg    = beta1_t * exp_avg    + (1.0 - beta1_t) * g
            exp_avg_sq = beta2_t * exp_avg_sq + (1.0 - beta2_t) * tf.square(g)

            denom = tf.sqrt(exp_avg_sq / bias_corr2) + tf.cast(self.eps, tf.float32)
            p = p - lr * (exp_avg / bias_corr1) / denom

            self._set_param_fp32(variable, idx, p)
            self._store('exp_avg',    idx, exp_avg,    variable.dtype)
            self._store('exp_avg_sq', idx, exp_avg_sq, variable.dtype)
            
    def save_own_variables(self, store) -> None:
        if not self.built:
            warnings.warn(
                f"Optimizer '{self.name}' has not been built; nothing to save.",
                stacklevel=2,
            )
            return

        store['__iteration__'] = self._iterations.numpy()
        if hasattr(self._learning_rate, 'numpy'):
            store['__learning_rate__'] = float(self._learning_rate.numpy())

        for i in range(len(self._trainable_variables)):
            if self.quantize and not self.compress_state_dict:
                store[f'exp_avg_{i}']    = self._materialize('exp_avg',    i).numpy()
                store[f'exp_avg_sq_{i}'] = self._materialize('exp_avg_sq', i).numpy()
            elif self.quantize:
                store[f'q_exp_avg_{i}']    = self.q_exp_avg[i].numpy()
                store[f's_exp_avg_{i}']    = self.s_exp_avg[i].numpy()
                store[f'q_exp_avg_sq_{i}'] = self.q_exp_avg_sq[i].numpy()
                store[f's_exp_avg_sq_{i}'] = self.s_exp_avg_sq[i].numpy()
            else:
                store[f'exp_avg_{i}']    = self.exp_avg[i].numpy()
                store[f'exp_avg_sq_{i}'] = self.exp_avg_sq[i].numpy()

            if self._error_bits[i] is not None:
                store[f'error_bits_{i}'] = self.error_bits[i].numpy()

    def load_own_variables(self, store) -> None:
        if not self.built:
            warnings.warn(
                f"Optimizer '{self.name}' has not been built; cannot load state.",
                stacklevel=2,
            )
            return

        if '__iteration__' in store:
            self._iterations.assign(int(store['__iteration__']))
        if '__learning_rate__' in store and hasattr(self._learning_rate, 'assign'):
            self._learning_rate.assign(float(store['__learning_rate__']))

        n = len(self._trainable_variables)
        for i in range(n):
            has_compressed  = f'q_exp_avg_{i}' in store
            has_uncompressed = f'exp_avg_{i}' in store

            if self.quantize:
                if has_compressed:
                    self.q_exp_avg[i].assign(
                        tf.cast(store[f'q_exp_avg_{i}'], tf.int8)
                    )
                    self.s_exp_avg[i].assign(
                        tf.cast(store[f's_exp_avg_{i}'], tf.float16)
                    )
                    self.q_exp_avg_sq[i].assign(
                        tf.cast(store[f'q_exp_avg_sq_{i}'], tf.uint8)
                    )
                    self.s_exp_avg_sq[i].assign(
                        tf.cast(store[f's_exp_avg_sq_{i}'], tf.float16)
                    )
                elif has_uncompressed:
                    q, s = quantize_state(
                        tf.constant(store[f'exp_avg_{i}'], dtype=tf.float32),
                        *_STATE_SPECS['exp_avg'],
                    )
                    self.q_exp_avg[i].assign(q)
                    self.s_exp_avg[i].assign(s)

                    q, s = quantize_state(
                        tf.constant(store[f'exp_avg_sq_{i}'], dtype=tf.float32),
                        *_STATE_SPECS['exp_avg_sq'],
                    )
                    self.q_exp_avg_sq[i].assign(q)
                    self.s_exp_avg_sq[i].assign(s)
                else:
                    warnings.warn(
                        f"No state found for parameter {i} in checkpoint; "
                        "moments left at zero.", stacklevel=2,
                    )
            else:
                if has_uncompressed:
                    self.exp_avg[i].assign(store[f'exp_avg_{i}'])
                    self.exp_avg_sq[i].assign(store[f'exp_avg_sq_{i}'])
                elif has_compressed:
                    self.exp_avg[i].assign(
                        dequantize_state(
                            tf.constant(store[f'q_exp_avg_{i}'], dtype=tf.int8),
                            tf.constant(store[f's_exp_avg_{i}'], dtype=tf.float16),
                            *_STATE_SPECS['exp_avg'],
                        )
                    )
                    self.exp_avg_sq[i].assign(
                        dequantize_state(
                            tf.constant(store[f'q_exp_avg_sq_{i}'], dtype=tf.uint8),
                            tf.constant(store[f's_exp_avg_sq_{i}'], dtype=tf.float16),
                            *_STATE_SPECS['exp_avg_sq'],
                        )
                    )
                else:
                    warnings.warn(
                        f"No state found for parameter {i} in checkpoint; "
                        "moments left at zero.", stacklevel=2,
                    )

            if self.error_bits[i] is not None and f'error_bits_{i}' in store:
                self.error_bits[i].assign(store[f'error_bits_{i}'])

    def get_fp32_model_state_dict(self, model) -> Dict[str, 'np.ndarray']:
        if not self.built:
            raise RuntimeError(
                'get_fp32_model_state_dict() requires the optimizer to be built. '
                'Call optimizer.build() or run at least one training step first.'
            )

        result: Dict[str, 'np.ndarray'] = {}
        for var in model.trainable_variables:
            name = var.path if hasattr(var, 'path') else var.name
            key = self._var_key(var)
            if key not in self._trainable_variables_indices:
                result[name] = tf.cast(var, tf.float32).numpy()
                continue
            idx = self._trainable_variables_indices[key]
            result[name] = self._get_param_fp32(var, idx).numpy()
        return result

    def set_fp32_model_state_dict(
        self, model, state_dict: Dict[str, 'np.ndarray']
    ) -> None:
        if not self.built:
            raise RuntimeError(
                'set_fp32_model_state_dict() requires the optimizer to be built. '
                'Call optimizer.build() or run at least one training step first.'
            )

        for var in model.trainable_variables:
            name = var.path if hasattr(var, 'path') else var.name
            if name not in state_dict:
                continue

            value_fp32 = tf.cast(
                tf.constant(state_dict[name]), tf.float32
            )
            key = self._var_key(var)
            if key not in self._trainable_variables_indices:
                var.assign(tf.cast(value_fp32, var.dtype))
                continue

            idx = self._trainable_variables_indices[key]

            if (
                self.error_bits[idx] is None
                and self.master_byte_width > 0
                and var.dtype in (tf.float16, tf.bfloat16)
            ):
                error_bytes = self.master_byte_width - _TF_DTYPE_BYTES[var.dtype]
                if error_bytes > 0:
                    error_dtype = tf.int8 if error_bytes == 1 else tf.int16
                    ecc_var = tf.Variable(
                        tf.zeros(var.shape, dtype=error_dtype),
                        trainable=False,
                        name=f'{self.name}/error_bits',
                    )
                    self.error_bits[idx] = ecc_var
                    self._track_variable(ecc_var)

            self._set_param_fp32(var, idx, value_fp32)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'beta1':              self.beta1,
            'beta2':              self.beta2,
            'eps':                self.eps,
            'quantize':           self.quantize,
            'master_weight_bits': self._master_weight_bits,
            'maximize':           self.maximize,
        })
        return config


class FlashAdamW_e(optimizer.Optimizer):
    """Enhanced FlashAdamW with compressed states and a full feature set.

    Args:
        learning_rate (float): Learning rate. Default ``1e-3``.
        beta1 (float): First-moment decay. Default ``0.9``.
        beta2 (float): Second-moment decay (also used by PNM noise norm).
            Default ``0.999``.
        eps (float): Adam epsilon. Default ``1e-8``.
        weight_decay (float): Weight-decay strength. Default ``1e-2``.
        weight_decouple (bool): Apply WD to the fp32 master weight
            multiplicatively (AdamW style) rather than additively into the
            gradient. Default ``True``.
        fixed_decay (bool): When ``weight_decouple=True``, apply
            ``p *= 1 - wd`` without scaling by ``lr``. Default ``False``.
        quantize (bool): Compress Adam moments to grouped 8-bit.
            Default ``True``.
        compress_state_dict (bool): Save raw int8/uint8 tensors in
            checkpoints. Set ``False`` for portable float32 checkpoints.
            Default ``True``.
        master_weight_bits (int | None): Master-weight precision for bf16/fp16
            params. One of ``{None, 24, 32}``. Default ``None``.
        maximize (bool): Maximise the objective. Default ``False``.
        orthograd (bool): Orthogonalise gradients w.r.t. their weights before
            the Adam step (handled by ``_backend_update_step``).
            Default ``False``.
        agc (bool): Unit-wise Adaptive Gradient Clipping. Default ``False``.
        agc_clip_val (float): AGC clipping ratio. Default ``1e-2``.
        agc_eps (float): AGC minimum weight norm. Default ``1e-3``.
        gc (bool): Gradient Centralization — subtract per-filter mean
            before the Adam step. Default ``False``.
        pnm (bool): Replace the first-moment EMA with Positive-Negative
            Momentum. Skips the quantised exp_avg buffers entirely.
            Default ``False``.
        cautious (bool): Mask the update to directions that agree with the
            gradient (Cautious-Adam). Default ``False``.
        trust_ratio (bool): Scale the update by ``‖w‖ / ‖update‖``
            (LARS-style). Default ``False``.
        trust_clip (bool): Clip trust ratio to ≤ 1. Default ``False``.
        lookahead (bool): Enable Lookahead slow-weight interpolation.
            Default ``False``.
        lookahead_merge_time (int): Lookahead sync frequency. Default ``5``.
        lookahead_blending_alpha (float): Lookahead blend factor. Default ``0.5``.
        name (str): Optimizer name.
    """

    _BITS_TO_BYTES: dict = {None: 0, 24: 3, 32: 4}
    _VALID_MASTER_WEIGHT_BITS: tuple = (None, 24, 32)

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        quantize: bool = True,
        compress_state_dict: bool = True,
        master_weight_bits: Optional[int] = None,
        maximize: bool = False,
        orthograd: bool = False,
        agc: bool = False,
        agc_clip_val: float = 1e-2,
        agc_eps: float = 1e-3,
        gc: bool = False,
        pnm: bool = False,
        cautious: bool = False,
        trust_ratio: bool = False,
        trust_clip: bool = False,
        lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        name: str = 'FlashAdamW_e',
        **kwargs,
    ):
        if master_weight_bits not in self._VALID_MASTER_WEIGHT_BITS:
            raise ValueError(
                f'master_weight_bits must be one of {self._VALID_MASTER_WEIGHT_BITS}; '
                f'got {master_weight_bits!r}'
            )

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.initial_lr: float = float(learning_rate)

        self.quantize = quantize
        self.compress_state_dict = compress_state_dict
        self._master_weight_bits = master_weight_bits
        self.master_byte_width = self._BITS_TO_BYTES[master_weight_bits]
        self.maximize = maximize

        self.orthograd = orthograd
        self.agc = agc
        self.agc_clip_val = agc_clip_val
        self.agc_eps = agc_eps
        self.gc = gc
        self.pnm = pnm
        self.cautious = cautious
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.lookahead = lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha

    def build(self, var_list: list) -> None:
        if self.built:
            return
        super().build(var_list)

        self.exp_avg: List = []

        self.q_exp_avg:    List[tf.Variable] = []
        self.s_exp_avg:    List[tf.Variable] = []

        self.q_exp_avg_sq: List[tf.Variable] = []
        self.s_exp_avg_sq: List[tf.Variable] = []

        self.error_bits: List[Optional[tf.Variable]] = []

        for var in var_list:
            numel: int = var.shape.num_elements() or int(tf.size(var).numpy())
            num_groups: int = max(math.ceil(numel / GROUP_SIZE), 1)

            if not self.pnm:
                if self.quantize:
                    q = tf.Variable(tf.zeros(var.shape, dtype=tf.int8),
                                    trainable=False, name='q_exp_avg')
                    s = tf.Variable(tf.zeros([num_groups], dtype=tf.float16),
                                    trainable=False, name='s_exp_avg')
                    self.q_exp_avg.append(q)
                    self.s_exp_avg.append(s)
                    self._track_variable(q)
                    self._track_variable(s)
                else:
                    self.exp_avg.append(
                        self.add_variable_from_reference(var, name='exp_avg')
                    )

            if self.quantize:
                q2 = tf.Variable(tf.zeros(var.shape, dtype=tf.uint8),
                                  trainable=False, name='q_exp_avg_sq')
                s2 = tf.Variable(tf.zeros([num_groups], dtype=tf.float16),
                                  trainable=False, name='s_exp_avg_sq')
                self.q_exp_avg_sq.append(q2)
                self.s_exp_avg_sq.append(s2)
                self._track_variable(q2)
                self._track_variable(s2)
            else:
                self.exp_avg_sq.append(
                    self.add_variable_from_reference(var, name='exp_avg_sq')
                )

            var_bytes = _TF_DTYPE_BYTES.get(var.dtype, 0)
            error_bytes = (
                self.master_byte_width - var_bytes
                if var_bytes and var.dtype in (tf.float16, tf.bfloat16)
                else 0
            )
            if error_bytes > 0:
                ecc = tf.Variable(
                    tf.zeros(var.shape, dtype=tf.int8 if error_bytes == 1 else tf.int16),
                    trainable=False, name='error_bits',
                )
                self.error_bits.append(ecc)
                self._track_variable(ecc)
            else:
                self.error_bits.append(None)

    def _materialize(self, name: str, idx: int) -> tf.Tensor:
        """De-quantize (or read) a moment variable as float32."""
        signed, sqrt, softsign = _STATE_SPECS[name]
        if self.quantize:
            if name == 'exp_avg':
                return dequantize_state(self.q_exp_avg[idx], self.s_exp_avg[idx],
                                        signed=signed, sqrt=sqrt, softsign=softsign)
            else:
                return dequantize_state(self.q_exp_avg_sq[idx], self.s_exp_avg_sq[idx],
                                        signed=signed, sqrt=sqrt, softsign=softsign)
        else:
            raw = self.exp_avg[idx] if name == 'exp_avg' else self.exp_avg_sq[idx]
            return tf.cast(raw, tf.float32)

    def _store(self, name: str, idx: int, value: tf.Tensor, var_dtype: tf.DType) -> None:
        """Quantize (or write) an updated moment variable."""
        signed, sqrt, softsign = _STATE_SPECS[name]
        if self.quantize:
            q, s = quantize_state(value, signed=signed, sqrt=sqrt, softsign=softsign)
            if name == 'exp_avg':
                self.q_exp_avg[idx].assign(q)
                self.s_exp_avg[idx].assign(s)
            else:
                self.q_exp_avg_sq[idx].assign(q)
                self.s_exp_avg_sq[idx].assign(s)
        else:
            target = self.exp_avg[idx] if name == 'exp_avg' else self.exp_avg_sq[idx]
            target.assign(tf.cast(value, var_dtype))

    def _get_param_fp32(self, var: tf.Variable, idx: int) -> tf.Tensor:
        """Return float32 view of parameter, applying ECC correction if present."""
        ecc = self.error_bits[idx]
        return reconstruct_fp32_param(var, ecc) if ecc is not None else tf.cast(var, tf.float32)

    def _set_param_fp32(self, var: tf.Variable, idx: int, value: tf.Tensor) -> None:
        """Write fp32 value back to parameter and refresh ECC bits."""
        var.assign(tf.cast(value, var.dtype))
        ecc = self.error_bits[idx]
        if ecc is not None:
            ecc.assign(compute_ecc_bits(value, var, self.master_byte_width))

    def update_step(self, grads: list, variables: list, learning_rate) -> None:
        lr = tf.cast(learning_rate, tf.float32)
        step = tf.cast(self._iterations + 1, tf.float32)
        beta1_t = tf.constant(self.beta1, dtype=tf.float32)
        beta2_t = tf.constant(self.beta2, dtype=tf.float32)
        bias_corr1 = 1.0 - beta1_t ** step
        bias_corr2 = 1.0 - beta2_t ** step
        eps_t = tf.constant(self.eps, dtype=tf.float32)

        for gradient, variable in zip(grads, variables):
            if gradient is None:
                continue

            idx = self._get_variable_index(variable)
            g = tf.cast(gradient, tf.float32)

            if self.maximize:
                g = -g

            if self.gc:
                g = self.gradient_centralize(g)

            if self.agc:
                g = self.apply_agc(
                    variable, g,
                    agc_eps=self.agc_eps,
                    agc_clip_val=self.agc_clip_val,
                )

            p = self._get_param_fp32(variable, idx)

            gradient = self.apply_weight_decay(variable, gradient, lr, 1.0 / self.initial_lr)

            exp_avg_sq = self._materialize('exp_avg_sq', idx)
            exp_avg_sq = beta2_t * exp_avg_sq + (1.0 - beta2_t) * tf.square(g)
            denom = tf.sqrt(exp_avg_sq / bias_corr2) + eps_t

            if self.pnm:
                first_moment = self.apply_pnm(g, step, idx)
                update = first_moment / denom
            else:
                exp_avg = self._materialize('exp_avg', idx)
                exp_avg = beta1_t * exp_avg + (1.0 - beta1_t) * g
                update = (exp_avg / bias_corr1) / denom
                self._store('exp_avg', idx, exp_avg, variable.dtype)

            if self.trust_ratio:
                update = self.apply_trust_ratio(variable, update)

            if self.cautious:
                update = self.apply_cautious(update, g)

            p = p - lr * update

            self._set_param_fp32(variable, idx, p)
            self._store('exp_avg_sq', idx, exp_avg_sq, variable.dtype)

            if self.lookahead:
                self.lookahead_merge(variable, step)

    def save_own_variables(self, store) -> None:
        if not self.built:
            warnings.warn(f"Optimizer '{self.name}' not built; nothing to save.", stacklevel=2)
            return

        store['__iteration__'] = self._iterations.numpy()
        if hasattr(self._learning_rate, 'numpy'):
            store['__learning_rate__'] = float(self._learning_rate.numpy())

        for i in range(len(self._trainable_variables)):
            if not self.pnm:
                if self.quantize and not self.compress_state_dict:
                    store[f'exp_avg_{i}'] = self._materialize('exp_avg', i).numpy()
                elif self.quantize:
                    store[f'q_exp_avg_{i}'] = self.q_exp_avg[i].numpy()
                    store[f's_exp_avg_{i}'] = self.s_exp_avg[i].numpy()
                else:
                    store[f'exp_avg_{i}'] = self.exp_avg[i].numpy()

            if self.quantize and not self.compress_state_dict:
                store[f'exp_avg_sq_{i}'] = self._materialize('exp_avg_sq', i).numpy()
            elif self.quantize:
                store[f'q_exp_avg_sq_{i}'] = self.q_exp_avg_sq[i].numpy()
                store[f's_exp_avg_sq_{i}'] = self.s_exp_avg_sq[i].numpy()
            else:
                store[f'exp_avg_sq_{i}'] = self.exp_avg_sq[i].numpy()

            if self.error_bits[i] is not None:
                store[f'error_bits_{i}'] = self.error_bits[i].numpy()

    def load_own_variables(self, store) -> None:
        if not self.built:
            warnings.warn(f"Optimizer '{self.name}' not built; cannot load.", stacklevel=2)
            return

        if '__iteration__' in store:
            self._iterations.assign(int(store['__iteration__']))
        if '__learning_rate__' in store and hasattr(self._learning_rate, 'assign'):
            self._learning_rate.assign(float(store['__learning_rate__']))

        for i in range(len(self._trainable_variables)):
            self._load_moment('exp_avg', i, store)
            self._load_moment('exp_avg_sq', i, store)

            if self.error_bits[i] is not None and f'error_bits_{i}' in store:
                self.error_bits[i].assign(store[f'error_bits_{i}'])

    def _load_moment(self, name: str, i: int, store) -> None:
        if name == 'exp_avg' and self.pnm:
            return

        has_compressed   = f'q_{name}_{i}' in store
        has_uncompressed = f'{name}_{i}' in store

        signed, sqrt, softsign = _STATE_SPECS[name]

        def _assign_quantized(q_val, s_val):
            q_dtype = tf.int8 if signed else tf.uint8
            if name == 'exp_avg':
                self.q_exp_avg[i].assign(tf.cast(q_val, q_dtype))
                self.s_exp_avg[i].assign(tf.cast(s_val, tf.float16))
            else:
                self.q_exp_avg_sq[i].assign(tf.cast(q_val, q_dtype))
                self.s_exp_avg_sq[i].assign(tf.cast(s_val, tf.float16))

        def _assign_float(fp32_val):
            target = (self.exp_avg[i] if name == 'exp_avg' else self.exp_avg_sq[i])
            target.assign(fp32_val)

        if self.quantize:
            if has_compressed:
                _assign_quantized(store[f'q_{name}_{i}'], store[f's_{name}_{i}'])
            elif has_uncompressed:
                q, s = quantize_state(
                    tf.constant(store[f'{name}_{i}'], dtype=tf.float32),
                    signed=signed, sqrt=sqrt, softsign=softsign,
                )
                _assign_quantized(q, s)
            else:
                warnings.warn(f"No state for '{name}[{i}]'; left at zero.", stacklevel=3)
        else:
            if has_uncompressed:
                _assign_float(store[f'{name}_{i}'])
            elif has_compressed:
                _assign_float(dequantize_state(
                    tf.constant(store[f'q_{name}_{i}'],
                                dtype=tf.int8 if signed else tf.uint8),
                    tf.constant(store[f's_{name}_{i}'], dtype=tf.float16),
                    signed=signed, sqrt=sqrt, softsign=softsign,
                ))
            else:
                warnings.warn(f"No state for '{name}[{i}]'; left at zero.", stacklevel=3)

    def get_fp32_model_state_dict(self, model) -> Dict[str, 'np.ndarray']:
        if not self.built:
            raise RuntimeError('Optimizer not built; call build() or run one step first.')
        result = {}
        for var in model.trainable_variables:
            vname = var.path if hasattr(var, 'path') else var.name
            key = self._var_key(var)
            if key not in self._trainable_variables_indices:
                result[vname] = tf.cast(var, tf.float32).numpy()
            else:
                idx = self._trainable_variables_indices[key]
                result[vname] = self._get_param_fp32(var, idx).numpy()
        return result

    def set_fp32_model_state_dict(
        self, model, state_dict: Dict[str, 'np.ndarray']
    ) -> None:
        if not self.built:
            raise RuntimeError('Optimizer not built; call build() or run one step first.')

        for var in model.trainable_variables:
            vname = var.path if hasattr(var, 'path') else var.name
            if vname not in state_dict:
                continue

            value_fp32 = tf.cast(tf.constant(state_dict[vname]), tf.float32)
            key = self._var_key(var)

            if key not in self._trainable_variables_indices:
                var.assign(tf.cast(value_fp32, var.dtype))
                continue

            idx = self._trainable_variables_indices[key]

            if (
                self.error_bits[idx] is None
                and self.master_byte_width > 0
                and var.dtype in (tf.float16, tf.bfloat16)
            ):
                error_bytes = self.master_byte_width - _TF_DTYPE_BYTES[var.dtype]
                if error_bytes > 0:
                    ecc = tf.Variable(
                        tf.zeros(var.shape,
                                 dtype=tf.int8 if error_bytes == 1 else tf.int16),
                        trainable=False, name='error_bits',
                    )
                    self.error_bits[idx] = ecc
                    self._track_variable(ecc)

            self._set_param_fp32(var, idx, value_fp32)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'beta1':                    self.beta1,
            'beta2':                    self.beta2,
            'eps':                      self.eps,
            'weight_decouple':          self.weight_decouple,
            'fixed_decay':              self.fixed_decay,
            'quantize':                 self.quantize,
            'compress_state_dict':      self.compress_state_dict,
            'master_weight_bits':       self._master_weight_bits,
            'maximize':                 self.maximize,
            'orthograd':                self.orthograd,
            'agc':                      self.agc,
            'agc_clip_val':             self.agc_clip_val,
            'agc_eps':                  self.agc_eps,
            'gc':                       self.gc,
            'pnm':                      self.pnm,
            'cautious':                 self.cautious,
            'trust_ratio':              self.trust_ratio,
            'trust_clip':               self.trust_clip,
            'lookahead':                self.lookahead,
            'lookahead_merge_time':     self.lookahead_merge_time,
            'lookahead_blending_alpha': self.lookahead_blending_alpha,
        })
        return config
