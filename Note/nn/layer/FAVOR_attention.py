import tensorflow as tf
import math


class FAVOR_attention:
    """Fast Attention Via positive Orthogonal Random features"""
    
    def __init__(
        self,
        key_dim,
        orthonormal=True,
        causal=False,
        m=128,
        redraw=True,
        h=None,
        f=[tf.nn.relu],
        randomizer=tf.random.normal,
        eps=0.0,
        kernel_eps=0.001,
        dtype='float32'
    ):
        self.key_dim = key_dim
        
        self.orthonormal = orthonormal
        self.causal = causal
        self.redraw = redraw
        self.m = m
        sqrt_m = math.sqrt(m)
        self.h = h if h is not None else lambda x: sqrt_m
        self.f = f
        self.randomizer = randomizer
        self.eps = eps
        self.kernel_eps = kernel_eps
        
        if orthonormal and m > key_dim:
            raise ValueError('m <= key_dim is required if orthonormal == True')
            
        self._features = None
        self.phi_scale = tf.cast(1. / sqrt_m, dtype=dtype)
        self.dtype=dtype

    def features(self):
        if self._features is None or self.redraw:
            self._features = self.randomizer(
                (self.key_dim, self.m),
                dtype=self.phi_scale.dtype
            )
            if self.orthonormal:
                self._features = tf.linalg.qr(
                    tf.cast(self._features, tf.float64))[0]
                self._features = tf.cast(self._features, self.phi_scale.dtype)
            self._features = tf.transpose(self._features)
        return self._features

    def __call__(self, keys, values, queries):
        """
        keys: (batch, keys_dimension, *keys_locations)
        values: (batch, values_dimension, *keys_locations)
        queries: (batch, keys_dimension, *queries_locations)
        """
        # flattening everything
        keys_locations = keys.shape[2:]
        queries_locations = queries.shape[2:]
        keys, values, queries = (tf.reshape(x, (*x.shape[:2], -1))
                                 for x in (keys, values, queries))
        
        if self.causal and keys_locations != queries_locations:
            raise ValueError(
                'Expected equal key and query locations with causal attention,'
                ' got: {}, {}'.format(keys_locations, queries_locations))
        
        # getting to (batch, n, dim)
        keys, values, queries = (tf.transpose(x, perm=[0, 2, 1])
                                 for x in (keys, values, queries))
        
        # features are (m, key_dim). randomized here if necessary
        features = self.features()
        
        # getting the randomized features for keys and queries
        def phi(x):
            # x is (batch, n, key_dim)

            # projections are (batch, n, m)
            projections = tf.matmul(x, tf.transpose(features))
            
            # (batch, n, r)
            return tf.concat(
                [f(projections) for f in self.f],
                axis=-1
            ) * self.h(x) * self.phi_scale + self.kernel_eps
                    
        # (batch, n_context, r)
        phi_k = phi(keys)
        # (batch, n, r)
        phi_q = phi(queries)
        
        if self.causal:
            # outer products of keys and values: (batch, n, r, dim)
            k_v_prod = tf.matmul(
                phi_k[:, :, :, tf.newaxis], values[:, :, tf.newaxis, :])

            out = tf.matmul(         # (batch, n, dim)
                phi_q[:, :, tf.newaxis, :],   # (batch, n, 1, r)
                tf.math.cumsum(k_v_prod, axis=1)  # (batch, n, r, dim)
            )[:, :, 0]
            
            # normalization factors: (batch, n, 1)
            norm = tf.matmul(
                phi_q[:, :, tf.newaxis, :],           # (batch, n, 1, r)
                tf.math.cumsum(phi_k, axis=1)[..., tf.newaxis]  # (batch, n, r, 1)
            )[:, :, 0]
        else:
            out = tf.matmul(  # (batch, n, dim)
                phi_q,
                tf.matmul(  # (batch, r, dim)
                    tf.transpose(phi_k, perm=[0, 2, 1]), values
                )
            )
            
            # normalization factors: (batch, n, 1)
            norm = tf.matmul(
                phi_q,
                tf.reduce_sum(phi_k, axis=1)[..., tf.newaxis]  # (batch, r, 1)
            )
        
        # normalizing
        out = out / (norm + 2 * self.eps * tf.cast(tf.abs(norm) <= self.eps, self.dtype))
        
        # restoring the desired shape
        out = tf.transpose(out, perm=[0, 2, 1])
        out = tf.reshape(out, (*out.shape[:2], *queries_locations))
        return out