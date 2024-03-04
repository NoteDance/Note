import tensorflow as tf

class RoPE:
    def __init__(self, dims: int, traditional: bool = False, base=None):
        self.dims = dims
        self.traditional = traditional
        self.base = base

    def _compute_rope(self, costheta, sintheta, x):
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = tf.concat([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = tf.concat([rx1, rx2], axis=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            raise NotImplementedError(
                "RoPE doesn't implement partial traditional application"
            )

        rx = tf.concat([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = tf.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=self.base, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return tf.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        dtype=tf.float32,
    ):
        D = D // 2
        positions = tf.range(offset, N, dtype=dtype)
        freqs = tf.math.exp(
            -tf.range(0, D, dtype=dtype) * (tf.math.log(base) / D)
        )
        theta = tf.reshape(positions, (-1, 1)) * tf.reshape(freqs, (1, -1))
        costheta = tf.math.cos(theta)
        sintheta = tf.math.sin(theta)

        return costheta, sintheta