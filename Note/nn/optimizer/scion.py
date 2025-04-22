""" SCION
https://arxiv.org/abs/2502.07529

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer
from enum import IntEnum


class LMONorm(IntEnum):
    r"""normalization types."""
    
    NONE = 0
    AUTO = 1
    SPECTRAL = 2
    SPECTRALCONV = 3
    SIGN = 4
    BIAS = 5
    COL = 6
    ROW = 7


def zero_power_via_newton_schulz_5(
    g, 
    num_steps = 5, 
    eps = 1e-7, 
    weights = (3.4445, -4.7750, 2.0315)
):
    
    if len(g.shape) != 2:
        raise ValueError('shape of g must be 2-dimensional')
    
    x = tf.cast(g, tf.bfloat16)
    norm_x = tf.norm(x)
    x = x / (norm_x + eps)
    
    shape = x.shape
    if shape[0] > shape[1]:
        x = tf.transpose(x)
    
    for _ in range(num_steps):
        a = tf.matmul(x, x, transpose_b=True)
        b = weights[1] * a + weights[2] * tf.matmul(a, a)
        x = weights[0] * x + tf.matmul(b, x)
    
    if g.shape[0] > g.shape[1]:
        x = tf.transpose(x)
    
    return x


class Norm:
    r"""Base class to perform norm onto Scion. This class does no norm."""
    
    def init(self, x):
        r"""Initialize parameter."""
        return x

    def lmo(self, grad):
        r"""Get LMO."""
        return grad


class Col(Norm):
    def __init__(self, normalized, transpose):
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x):
        dtype = x.dtype
        if self.transpose:
            x = tf.transpose(x, perm=[1, 0])
            
        x = tf.random.normal(shape=x.shape, dtype=x.dtype)
        
        norm_value = tf.norm(x, axis=0, keepdims=True)
        d0 = tf.cast(tf.shape(x)[0], x.dtype)
        x = x / (norm_value) * tf.math.sqrt(d0)
        if self.normalized:
            d1 = tf.cast(tf.shape(x)[1], x.dtype)
            x = x / d1
            
        x = tf.cast(x, dtype=dtype)
        if self.transpose:
            x = tf.transpose(x, perm=[1, 0])
        return x

    def lmo(self, grad, eps = 1e-8):
        if self.transpose:
            grad = tf.transpose(grad, perm=[1, 0])
            
        d_in = tf.shape(grad)[0]
        d_out = tf.shape(grad)[1]
        
        rms_value = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=0, keepdims=True)) / tf.math.sqrt(tf.cast(d_in, grad.dtype))
        if self.normalized:
            rms_value = rms_value * tf.cast(d_out, grad.dtype)
            
        grad = grad / (rms_value + eps)
        
        if self.transpose:
            grad = tf.transpose(grad, perm=[1, 0])
            
        return grad


class Row(Norm):
    def __init__(self, normalized, transpose):
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x):
        dtype = x.dtype
        if self.transpose:
            x = tf.transpose(x, perm=[1, 0])
            
        x = tf.random.normal(shape=x.shape, dtype=x.dtype)
        
        norm_value = tf.norm(x, axis=-1, keepdims=True)
        x = x / (norm_value)
        if self.normalized:
            d_last = tf.cast(tf.shape(x)[-1], x.dtype)
            x = x / tf.math.sqrt(d_last)
            
        x = tf.cast(x, dtype=dtype)
        if self.transpose:
            x = tf.transpose(x, perm=[1, 0])
            
        return x

    def lmo(self, grad, eps):
        if self.transpose:
            grad = tf.transpose(grad, perm=[1, 0])
            
        rms_value = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=-1, keepdims=True))
        if self.normalized:
            d_last = tf.cast(tf.shape(grad)[-1], grad.dtype)
            rms_value = rms_value * tf.math.sqrt(d_last)
            
        grad = grad / (rms_value + eps)
        
        if self.transpose:
            grad = tf.transpose(grad, perm=[1, 0])
            
        return grad


class BiasRMS(Norm):
    r"""bias RMS."""
    
    def init(self, x):
        return tf.zeros_like(x)

    def lmo(self, grad, eps = 1e-8):
        rms_value = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=0, keepdims=True))
        grad = grad / (rms_value + eps)
        return grad


class SpectralConv(Norm):
    def __init__(self, num_steps = 5):
        self.num_steps = num_steps

    def init(self, x):
        x_fp64 = tf.cast(x, tf.float64)
        shape = x_fp64.shape.as_list()
        
        d_out, d_in, kernel_size = shape[0], shape[1], shape[2]
        
        orthogonal_initializer = tf.keras.initializers.Orthogonal()
        rows = []
        for i in range(kernel_size):
            cols = []
            for j in range(kernel_size):
                slice_ij = x_fp64[:, :, i, j]
                new_slice = orthogonal_initializer(shape=slice_ij.shape)
                cols.append(new_slice)
            row_tensor = tf.stack(cols, axis=-1)
            rows.append(row_tensor)
        x_fp64_new = tf.stack(rows, axis=-2)
        
        scale = tf.math.sqrt(tf.cast(d_out, x.dtype) / tf.cast(d_in, x.dtype)) / (kernel_size ** 2)
        x_fp64_new = x_fp64_new * scale
        
        return tf.cast(x_fp64_new, x.dtype)

    def lmo(self, grad, eps = 1e-8):
        orig_shape = grad.shape
        grad_2d = tf.reshape(grad, [tf.shape(grad)[0], -1])
        grad_2d = zero_power_via_newton_schulz_5(grad_2d, self.num_steps)
        grad = tf.reshape(grad_2d, orig_shape)
        shape = grad.shape.as_list()
        d_out, d_in, kernel_size = shape[0], shape[1], shape[2]
        scale = tf.math.sqrt(tf.cast(d_out, grad.dtype) / tf.cast(d_in, grad.dtype)) / (kernel_size ** 2)
        grad = grad * scale
        return grad


class Spectral(Norm):
    def __init__(self, max_scale: bool = False, normalize: bool = True, num_steps: int = 5) -> None:
        self.max_scale = max_scale
        self.normalize = normalize
        self.num_steps = num_steps

    def init(self, x):
        x_fp64 = tf.cast(x, tf.float64)
        
        orthogonal_initializer = tf.keras.initializers.Orthogonal()
        new_x = orthogonal_initializer(shape=x_fp64.shape)
        x_fp64 = tf.convert_to_tensor(new_x, dtype=tf.float64)
        shape = x_fp64.shape.as_list()
        d_out, d_in = shape
        
        scale = tf.math.sqrt(tf.cast(d_out, x.dtype) / tf.cast(d_in, x.dtype)) if self.normalize else tf.math.sqrt(tf.cast(d_out, x.dtype))
        if self.max_scale:
            scale = tf.maximum(1.0, scale)
            
        x_fp64 = x_fp64 * scale
        
        return tf.cast(x_fp64, x.dtype)

    def lmo(self, grad, eps = 1e-8):
        orig_shape = grad.shape
        grad_2d = tf.reshape(grad, [tf.shape(grad)[0], -1])
        grad_2d = zero_power_via_newton_schulz_5(grad_2d, self.num_steps)
        grad = tf.reshape(grad_2d, orig_shape)
        
        d_out, d_in = grad.shape.as_list()
        
        scale = tf.math.sqrt(tf.cast(d_out, grad.dtype) / tf.cast(d_in, grad.dtype)) if self.normalize else tf.math.sqrt(tf.cast(d_out, grad.dtype))
        if self.max_scale:
            scale = tf.maximum(1.0, scale)
            
        grad = grad * scale
        return grad


class Sign(Norm):
    def __init__(self, zero_init: bool = False, normalize: bool = True) -> None:
        self.zero_init = zero_init
        self.normalize = normalize

    def init(self, x):
        if self.zero_init:
            return tf.zeros_like(x)
        
        d_in = tf.cast(tf.shape(x)[1], x.dtype)
        
        x = 2 * tf.cast(tf.random.uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32), x.dtype) - 1
        if self.normalize:
            x = x / d_in
            
        return x

    def lmo(self, grad: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
        d_in = tf.cast(tf.shape(grad)[1], grad.dtype)
        if self.normalize:
            return tf.sign(grad) / d_in
        else:
            return tf.sign(grad)


class Auto(Norm):
    r"""choose Norm type automatically."""
    
    def init(self, x):
        ndim = len(x.shape)
        if ndim in (0, 1):
            return BiasRMS().init(x)
        if ndim == 2:
            return Spectral().init(x)
        if ndim in (3, 4):
            return SpectralConv().init(x)
        raise NotImplementedError

    def lmo(self, grad):
        ndim = len(grad.shape)
        if ndim in (0, 1):
            return BiasRMS().lmo(grad)
        if ndim == 2:
            return Spectral().lmo(grad)
        if ndim in (3, 4):
            return SpectralConv().lmo(grad)
        raise NotImplementedError


def build_lmo_norm(norm_type, **kwargs):  # noqa: PLR0911
    r"""Build LMONorm by given norm_type."""
    if norm_type == LMONorm.AUTO:
        return Auto()
    if norm_type == LMONorm.SPECTRAL:
        return Spectral(**kwargs)
    if norm_type == LMONorm.SPECTRALCONV:
        return SpectralConv(**kwargs)
    if norm_type == LMONorm.SIGN:
        return Sign(**kwargs)
    if norm_type == LMONorm.BIAS:
        return BiasRMS()
    if norm_type == LMONorm.COL:
        return Col(**kwargs)
    if norm_type == LMONorm.ROW:
        return Row(**kwargs)
    return Norm()


class SCION(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=0.0,
        momentum=0.1,
        constraint=False,
        norm_type=LMONorm.AUTO,
        norm_kwargs=None,
        scale=1.0,
        weight_decouple=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="scion",
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
        self.constraint = constraint
        self.norm_type = norm_type
        self.norm_kwargs = norm_kwargs
        self.scale = scale
        self.weight_decouple = weight_decouple
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.momentum_buffer[self._get_variable_index(var)] = self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                                        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.momentum_buffer = []
        self.norm = build_lmo_norm(self.norm_type, **self.norm_kwargs)
        for var in var_list:
            self.momentum_buffer.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum_buffer"
                                        ))
        
    def init(self):
        norm = build_lmo_norm(self.norm_type, **self.norm_kwargs)
        for var in self._trainable_variables:
            norm.init(var)
            var.assign(var *self.scale)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'SCION does not support sparse gradients')

        if self.momentum != 1:
            momentum_buffer = self.momentum_buffer[self._get_variable_index(variable)]
            momentum_buffer.assign(momentum_buffer * (1.0 - self.momentum) + gradient * self.momentum)
            gradient = momentum_buffer

        update = self.norm.lmo(gradient) * self.scale

        if self.constraint:
            variable.assign(variable * (1.0 - lr))

        if not self.constraint and self.weight_decay > 0.0:
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                gradient += variable * self.weight_decay

        variable.assign_add(update * -lr)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "constraint": self.constraint,
                "norm_type": self.norm_type,
                "norm_kwargs": self.norm_kwargs,
                "scale": self.scale,
                "weight_decouple": self.weight_decouple,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


class SCIONLight(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=0.0,
        momentum=0.1,
        constraint=False,
        norm_type=LMONorm.AUTO,
        norm_kwargs=None,
        scale=1.0,
        weight_decouple=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="scionlight",
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
        self.constraint = constraint
        self.norm_type = norm_type
        self.norm_kwargs = norm_kwargs
        self.scale = scale
        self.weight_decouple = weight_decouple
    
    def reset(self):
        pass

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.norm = build_lmo_norm(self.norm_type, **self.norm_kwargs)
        
    def init(self):
        norm = build_lmo_norm(self.norm_type, **self.norm_kwargs)
        for var in self._trainable_variables:
            norm.init(var)
            var.assign(var *self.scale)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'SCIONLight does not support sparse gradients')

        update = self.norm.lmo(gradient) * self.scale

        if self.constraint:
            variable.assign(variable * (1.0 - lr))

        if not self.constraint and self.weight_decay > 0.0:
            if self.weight_decouple:
                variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
            elif self.weight_decay > 0.0:
                gradient += variable * self.weight_decay

        variable.assign_add(update * -lr)
        
        if self.momentum != 1.0:
            gradient[self._get_variable_index(variable)] *= (1.0 - self.momentum)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "constraint": self.constraint,
                "norm_type": self.norm_type,
                "norm_kwargs": self.norm_kwargs,
                "scale": self.scale,
                "weight_decouple": self.weight_decouple,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass