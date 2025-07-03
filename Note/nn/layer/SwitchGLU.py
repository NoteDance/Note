import tensorflow as tf
from Note import nn
import math

class SwitchLinear(nn.Layer):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = nn.Parameter(tf.random.uniform(
            minval=-scale,
            maxval=scale,
            shape=(num_experts, input_dims, output_dims),
        ))

        self.use_bias=bias
        if bias:
            self.bias = nn.Parameter(tf.zeros((num_experts, output_dims)))

    @property
    def input_dims(self):
        return self.weight.shape[1]

    @property
    def output_dims(self):
        return self.weight.shape[2]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices):
        x = nn.gather_mm(x, self.weight, indices)
        if self.use_bias:
            x = x + tf.expand_dims(tf.gather(self.bias, indices), -2)
        return x

class SwitchGLU(nn.Layer):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=tf.nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices):
        
        x_up = self.up_proj(x, indices)
        x_gate = self.gate_proj(x, indices)
        x = self.down_proj(self.activation(x_gate) * x_up, indices)

        return x