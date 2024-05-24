import tensorflow as tf
from Note import nn
import math

class SwitchLinear:
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        scale = math.sqrt(1 / input_dims)
        self.weight = tf.Variable(tf.random.uniform(
            minval=-scale,
            maxval=scale,
            shape=(num_experts, input_dims, output_dims),
        ))
        nn.Model.param.append(self.weight)

        self.use_bias=bias
        if bias:
            self.bias = tf.Variable(tf.zeros((num_experts, output_dims)))
            nn.Model.param.append(self.bias)

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

class SwitchGLU:
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=tf.nn.silu,
        bias: bool = False,
    ):

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices):
        
        x_up = self.up_proj(x, indices)
        x_gate = self.gate_proj(x, indices)
        x = self.down_proj(self.activation(x_gate) * x_up, indices)

        return x