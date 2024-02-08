import tensorflow as tf
from Note.nn.layer.dense import dense

class LoRALinear:
    @staticmethod
    def from_linear(linear, rank: int = 8):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self):
        linear = self.linear
        bias = linear.use_bias
        weight = linear.weight

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        output_dims, input_dims = weight.shape
        fused_linear = dense(output_dims, input_dims, bias=bias)

        lora_b = tf.cast((self.scale * tf.transpose(self.lora_b)), dtype)
        lora_a = tf.cast(tf.transpose(self.lora_a), dtype)
        fused_linear.weight = weight + tf.matmul(lora_b, lora_a)
        if bias:
            fused_linear.bias = linear.bias

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        lora_rank: int = 8,
        bias: bool = False,
        scale: float = 20.0,
    ):
        # Regular linear layer weights
        self.linear = dense(output_dims, input_dims, bias=bias)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / tf.math.sqrt(input_dims)
        self.lora_a = tf.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = tf.zeros(shape=(lora_rank, output_dims))

    def __call__(self, data):
        dtype = self.linear.weight.dtype
        y = self.linear(tf.cast(data, dtype))
        z = tf.matmul(tf.matmul(data, self.lora_a), self.lora_b)
        return y + self.scale * z