""" Note Conditionally Parameterized Convolution (CondConv)

Paper: CondConv: Conditionally Parameterized Convolutions for Efficient Inference
(https://arxiv.org/abs/1904.04971)

Hacked together by / Copyright 2025 NoteDance
"""

import tensorflow as tf
from Note import nn
import math
from functools import partial


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = math.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(tf.reshape(weight[i], (expert_shape)))
    return condconv_initializer


class CondConv2d(nn.Layer):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    
    def __init__(self, filters, kernel_size=3, input_size=None, strides=1, padding='', dilations=1, groups=1, use_bias=False, num_experts=4):
        super().__init__()
        self.output_size = filters
        self.kernel_size = nn.to_2tuple(kernel_size)
        self.input_size = input_size
        self.strides = nn.to_2tuple(strides)
        padding = ((strides - 1) + dilations * (kernel_size - 1)) // 2
        self.padding = nn.to_2tuple(padding)
        self.dilations = nn.to_2tuple(dilations)
        self.groups = groups
        self.use_bias = use_bias
        self.num_experts = num_experts
        self.init_weights=None
        
        self.weight_shape = self.kernel_size + (self.in_channels // self.groups, self.out_channels)
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight_num_param = weight_num_param
        if input_size!=None:
            self.weight = nn.Parameter(tf.zeros((self.num_experts, weight_num_param)))
            
            if use_bias:
                self.bias_shape = (self.out_channels,)
                self.bias = nn.Parameter(tf.zeros((self.num_experts, self.out_channels)))
            else:
                self.bias = None

            self.reset_parameters()
            
            nn.Model.param_dict['conv2d_weight'].append(self.weight)
            if use_bias==True:
                nn.Model.param_dict['conv2d_bias'].append(self.bias)
        
        if len(nn.Model.name_list)>0:
            nn.Model.name_=nn.Model.name_list[-1]
        if nn.Model.name_!=None and nn.Model.name_ not in nn.Model.layer_dict:
            nn.Model.layer_dict[nn.Model.name_]=[]
            nn.Model.layer_dict[nn.Model.name_].append(self)
        elif nn.Model.name_!=None:
            nn.Model.layer_dict[nn.Model.name_].append(self)
    
    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = math.prod(self.weight_shape[1:4])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)
    
    def build(self):
        self.weight = nn.Parameter(tf.zeros((self.num_experts, self.weight_num_param)))
        
        if self.use_bias:
            self.bias_shape = (self.out_channels,)
            self.bias = nn.Parameter(tf.zeros((self.num_experts, self.out_channels)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
        nn.Model.param_dict['conv2d_weight'].append(self.weight)
        if self.use_bias==True:
            nn.Model.param_dict['conv2d_bias'].append(self.bias)
            
        if self.init_weights!=None:
            self.init_weights(self)
        return
    
    def __call__(self, x, routing_weights):
        if x.dtype != self.weight.dtype:
            x = tf.cast(x, self.weight.dtype)
        if self.input_size == None:
            self.input_size = x.shape[-1]
            self.build()
            
        B, H, W, C = x.shape
        weight = tf.matmul(routing_weights, self.weight)
        new_weight_shape = self.kernel_size + (self.in_channels // self.groups, B * self.out_channels)
        weight = tf.reshape(weight, (new_weight_shape))
        bias = None
        if self.bias is not None:
            bias = tf.matmul(routing_weights, self.bias)
            bias = tf.reshape(bias, (B * self.out_channels))
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        # reshape instead of view to work with channels_last input
        x = tf.transpose(x, (1, 2, 0, 3))
        x = tf.reshape(x, (1, H, W, B * C))
        out = nn.conv2d_func(
            x, weight, bias, strides=self.stride, padding=self.padding,
            dilations=self.dilation, groups=self.groups * B)
        out = tf.reshape(out, (out.shape[1], out.shape[2], B, self.out_channels))
        out = tf.transpose(out, [2, 0, 1, 3])
        
        return out