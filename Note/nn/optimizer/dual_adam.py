import tensorflow as tf
from Note.nn.optimizer import optimizer


class DualAdam(optimizer.Optimizer):
    """Combining Adam and its inverse counterpart to enhance generalization.

    During early training, blends standard Adam with an "inverse Adam" update
    (multiplying by de_nom instead of dividing) according to a linearly decaying
    rate. Once inverse_adam_rate drops below switch_rate, falls back to plain Adam.

    Args:
        learning_rate (float): Learning rate. Default: 1e-3.
        beta1 (float): EMA coefficient for the first moment. Default: 0.9.
        beta2 (float): EMA coefficient for the second moment. Default: 0.999.
        switch_rate (float): Linear decay rate for the inverse Adam contribution.
            inverse_adam_rate = max(0, 1 - step * switch_rate). Default: 1e-2.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        weight_decouple (bool): Use decoupled weight decay (AdamW style). Default: False.
        fixed_decay (bool): Keep weight decay fixed (ignore lr scaling). Default: False.
        eps (float): Denominator epsilon for numerical stability. Default: 1e-8.
        maximize (bool): Maximize the objective instead of minimizing. Default: False.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        switch_rate: float = 1e-2,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        name: str = "dual_adam",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            **kwargs,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.switch_rate = switch_rate
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.eps = eps
        self.maximize = maximize

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        for var in var_list:
            self.exp_avg.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg"
                )
            )
            self.exp_avg_sq.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="exp_avg_sq"
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError("DualAdam does not support sparse gradients.")

        if self.maximize:
            gradient = -gradient

        lr = tf.cast(learning_rate, variable.dtype)

        # Coupled or decoupled weight decay (handled by BaseOptimizer helper)
        gradient = self.apply_weight_decay(variable, gradient, lr)

        idx = self._get_variable_index(variable)
        exp_avg     = self.exp_avg[idx]
        exp_avg_sq  = self.exp_avg_sq[idx]

        beta1 = tf.cast(self.beta1, variable.dtype)
        beta2 = tf.cast(self.beta2, variable.dtype)
        eps   = tf.cast(self.eps,   variable.dtype)

        # Step is 1-based (iterations is incremented *after* update_step)
        step = tf.cast(self.iterations + 1, variable.dtype)

        bias_correction1 = 1.0 - tf.pow(beta1, step)
        bias_correction2 = 1.0 - tf.pow(beta2, step)

        # inverse_adam_rate mirrors PyTorch: max(0, 1 - step * switch_rate)
        switch_rate      = tf.cast(self.switch_rate, variable.dtype)
        inverse_adam_rate = tf.maximum(
            tf.zeros([], dtype=variable.dtype),
            1.0 - step * switch_rate,
        )
        use_inverse_adam = inverse_adam_rate >= switch_rate

        # Update EMA moments
        exp_avg.assign(exp_avg * beta1 + gradient * (1.0 - beta1))
        exp_avg_sq.assign(exp_avg_sq * beta2 + gradient * gradient * (1.0 - beta2))

        exp_avg_hat = exp_avg / bias_correction1
        de_nom      = tf.sqrt(exp_avg_sq / bias_correction2) + eps

        # Choose update rule
        def _inverse_adam():
            # Blend: weight = (1/de_nom)*(1-r) + de_nom*r
            blend = (1.0 / de_nom) * (1.0 - inverse_adam_rate) + de_nom * inverse_adam_rate
            return exp_avg_hat * blend

        def _standard_adam():
            return exp_avg_hat / de_nom

        update = tf.cond(use_inverse_adam, _inverse_adam, _standard_adam)

        variable.assign_add(-lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1":          self.beta1,
                "beta2":          self.beta2,
                "switch_rate":    self.switch_rate,
                "weight_decouple": self.weight_decouple,
                "fixed_decay":    self.fixed_decay,
                "eps":            self.eps,
                "maximize":       self.maximize,
            }
        )
        return config


class DualAdam_e(optimizer.Optimizer):
    """Enhanced DualAdam with the full BaseOptimizer feature set.

    Combines Adam and its inverse counterpart (DualAdam) with optional:
    - Orthogonal gradient projection  (orthograd)
    - Adaptive Gradient Clipping      (enable_agc)
    - Cautious update masking         (cautious)
    - Lookahead meta-optimizer        (lookahead)
    - Positive-Negative Momentum      (pnm)  — replaces first moment EMA
    - Layer-wise trust-ratio scaling  (trust_ratio / trust_clip)

    DualAdam core logic
    -------------------
    inverse_adam_rate = max(0, 1 - step * switch_rate)
    use_inverse_adam  = inverse_adam_rate >= switch_rate

    If use_inverse_adam:
        blend  = (1/de_nom) * (1 - r) + de_nom * r     # r = inverse_adam_rate
        update = exp_avg_hat * blend
    Else (standard Adam):
        update = exp_avg_hat / de_nom

    Args:
        learning_rate (float): Learning rate. Default: 1e-3.
        beta1 (float): EMA coefficient for the first moment. Default: 0.9.
        beta2 (float): EMA coefficient for the second moment. Default: 0.999.
        switch_rate (float): Decay rate for the inverse Adam contribution.
            inverse_adam_rate = max(0, 1 - step * switch_rate). Default: 1e-2.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        weight_decouple (bool): Decoupled weight decay (AdamW). Default: False.
        fixed_decay (bool): Fixed (non-lr-scaled) weight decay. Default: False.
        eps (float): Denominator epsilon. Default: 1e-8.
        maximize (bool): Maximize the objective. Default: False.
        orthograd (bool): Project gradients to be orthogonal to weights. Default: False.
        enable_agc (bool): Adaptive Gradient Clipping per parameter. Default: False.
        agc_eps (float): AGC minimum weight norm floor. Default: 1e-3.
        agc_clip_val (float): AGC clipping ratio. Default: 1e-2.
        cautious (bool): Mask update steps misaligned with gradient sign. Default: False.
        lookahead (bool): Lookahead meta-optimizer. Default: False.
        lookahead_merge_time (int): Steps between slow-weight merges. Default: 5.
        lookahead_blending_alpha (float): Slow-weight interpolation factor. Default: 0.5.
        pnm (bool): Positive-Negative Momentum (replaces exp_avg). Default: False.
        trust_ratio (bool): Layer-wise trust ratio (LARS-style). Default: False.
        trust_clip (bool): Clip trust ratio to ≤ 1.0. Default: False.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        switch_rate: float = 1e-2,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        orthograd: bool = False,
        agc: bool = False,
        agc_eps: float = 1e-3,
        agc_clip_val: float = 1e-2,
        cautious: bool = False,
        lookahead: bool = False,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        pnm: bool = False,
        trust_ratio: bool = False,
        trust_clip: bool = False,
        sn: bool = False,
        subset_size: int = -1,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name: str = "dual_adam_e",
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.switch_rate = switch_rate
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.eps = eps
        self.maximize = maximize
        self.orthograd = orthograd
        self.agc = agc
        self.agc_eps = agc_eps
        self.agc_clip_val = agc_clip_val
        self.cautious = cautious
        self.lookahead = lookahead
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.pnm = pnm
        self.trust_ratio = trust_ratio
        self.trust_clip = trust_clip
        self.sn = sn
        self.subset_size = subset_size

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg    = []
        if not self.sn:
            self.exp_avg_sq = []
        self.slow_momentum = []
        self.pos_momentum  = []
        self.neg_momentum  = []

        if self.use_ema:
            self._model_variables_moving_average = self.add_optimizer_variables(
                var_list, "average"
            )
        if self.gradient_accumulation_steps:
            self._accumulated_gradients = []

        for i, var in enumerate(var_list):
            self._trainable_variables_indices[self._var_key(var)] = i

            if self.gradient_accumulation_steps:
                self._accumulated_gradients.append(
                    self.add_variable_from_reference(
                        var, name="gradient_accumulator"
                    )
                )

            # Second moment — always required
            if not self.sn:
                self.exp_avg_sq.append(
                    self.add_variable_from_reference(var, name="exp_avg_sq")
                )

            # First moment: PNM uses two buffers; standard Adam uses one
            if self.pnm:
                self.exp_avg.append(None)  # unused placeholder; keeps index aligned
                self.pos_momentum.append(
                    self.add_variable_from_reference(var, name="pos_momentum")
                )
                self.neg_momentum.append(
                    self.add_variable_from_reference(var, name="neg_momentum")
                )
            else:
                self.exp_avg.append(
                    self.add_variable_from_reference(var, name="exp_avg")
                )

            # Lookahead slow weights
            if self.lookahead:
                self.slow_momentum.append(tf.Variable(var))
                self._track_variable(self.slow_momentum[-1])

        self._trainable_variables = var_list[:]
        self.built = True

    def update_step(self, grads, trainable_variables, learning_rate):
        # Orthogonal gradient projection (modifies the grads list in-place)
        if self.orthograd:
            self.apply_orthogonal_gradients(trainable_variables, grads)

        for p, g in zip(trainable_variables, grads):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError(
                    "DualAdam_e does not support sparse gradients."
                )

            grad = -g if self.maximize else g
            lr   = tf.cast(learning_rate, p.dtype)
            step = tf.cast(self.iterations + 1, p.dtype)

            # ── Adaptive Gradient Clipping ─────────────────────────────────
            if self.agc:
                grad = self.apply_agc(
                    p, grad,
                    agc_eps=self.agc_eps,
                    agc_clip_val=self.agc_clip_val,
                )

            # ── Weight decay (coupled or decoupled) ────────────────────────
            grad = self.apply_weight_decay(p, grad, lr)

            idx        = self._get_variable_index(p)
            exp_avg_sq = self.exp_avg_sq[idx]
            beta1      = tf.cast(self.beta1, p.dtype)
            beta2      = tf.cast(self.beta2, p.dtype)
            eps        = tf.cast(self.eps,   p.dtype)

            bias_correction1 = 1.0 - tf.pow(beta1, step)
            bias_correction2 = 1.0 - tf.pow(beta2, step)

            # ── Dual-Adam decay rate ───────────────────────────────────────
            switch_rate       = tf.cast(self.switch_rate, p.dtype)
            inverse_adam_rate = tf.maximum(
                tf.zeros([], dtype=p.dtype),
                1.0 - step * switch_rate,
            )
            use_inverse_adam = inverse_adam_rate >= switch_rate

            # ── Second moment update ───────────────────────────────────────
            if self.sn:
                second_moment = self.get_second_moment_update(grad, idx)
                exp_avg_sq.assign(
                    exp_avg_sq * beta2 + second_moment * (1.0 - beta2)
                )
            else:
                exp_avg_sq.assign(
                    exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
                )
            de_nom = tf.sqrt(exp_avg_sq / bias_correction2) + eps

            # ── First moment / gradient direction ──────────────────────────
            # PNM: alternates positive/negative momentum buffers to produce a
            # lower-variance gradient estimate, replacing exp_avg entirely.
            # exp_avg_sq is still used for the denominator below.
            if self.pnm:
                exp_avg_hat = self.apply_pnm(grad, step, idx)
            else:
                exp_avg = self.exp_avg[idx]
                exp_avg.assign(exp_avg * beta1 + grad * (1.0 - beta1))
                exp_avg_hat = exp_avg / bias_correction1
            
            # ── Reshape numerator for subset-norm compatibility ────────────
            # After reshaping: exp_avg_hat (n_subsets, ss)
            #                  de_nom      (n_subsets, 1)   ← broadcasts over ss
            if self.sn:
                exp_avg_hat = self.get_reshaped_exg_avg(exp_avg, grad, idx)

            # ── Dual Adam update ───────────────────────────────────────────
            # Inverse Adam blends (1/de_nom) with de_nom, interpolated by r.
            # As training progresses and r → 0, this collapses to standard Adam.
            def _inverse_adam():
                blend = (
                    (1.0 / de_nom) * (1.0 - inverse_adam_rate)
                    + de_nom       *         inverse_adam_rate
                )
                result = exp_avg_hat * blend
                if self.sn:
                    result = tf.reshape(result, p.shape)
                return result

            def _standard_adam():
                result = exp_avg_hat / de_nom
                if self.sn:
                    result = tf.reshape(result, p.shape)
                return result

            update = tf.cond(use_inverse_adam, _inverse_adam, _standard_adam)

            # ── Trust ratio (LARS-style layer-wise scaling) ────────────────
            if self.trust_ratio:
                update = self.apply_trust_ratio(p, update)

            # ── Cautious mask ──────────────────────────────────────────────
            # Zero-out update elements whose sign disagrees with the gradient;
            # rescale the rest to preserve the expected update magnitude.
            if self.cautious:
                update = self.apply_cautious(update, grad)

            # ── Parameter step ─────────────────────────────────────────────
            p.assign_add(-lr * update)

            # ── Lookahead slow-weight merge ────────────────────────────────
            if self.lookahead:
                self.lookahead_merge(p, step)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1":                    self.beta1,
                "beta2":                    self.beta2,
                "switch_rate":              self.switch_rate,
                "weight_decouple":          self.weight_decouple,
                "fixed_decay":              self.fixed_decay,
                "eps":                      self.eps,
                "maximize":                 self.maximize,
                "orthograd":                self.orthograd,
                "agc":                      self.agc,
                "agc_eps":                  self.agc_eps,
                "agc_clip_val":             self.agc_clip_val,
                "cautious":                 self.cautious,
                "lookahead":                self.lookahead,
                "lookahead_merge_time":     self.lookahead_merge_time,
                "lookahead_blending_alpha": self.lookahead_blending_alpha,
                "pnm":                      self.pnm,
                "trust_ratio":              self.trust_ratio,
                "trust_clip":               self.trust_clip,
                "sn":                       self.sn,
                "subset_size":              self.subset_size,
            }
        )
        return config