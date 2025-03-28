""" Note Lamb optimizer w/ behaviour similar to NVIDIA FusedLamb

This optimizer code was adapted from the following (starting with latest)
* https://github.com/HabanaAI/Model-References/blob/2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
* https://github.com/cybertronai/pytorch-lamb

Use FusedLamb if you can (GPU). The reason for including this variant of Lamb is to have a version that is
similar in behaviour to APEX FusedLamb if you aren't using NVIDIA GPUs or cannot install/use APEX.

In addition to some cleanup, this Lamb impl has been modified to support PyTorch XLA and has been tested on TPU.

Original copyrights for above sources are below.

Modifications Copyright 2024 NoteDance
"""
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2019 cybertronai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import tensorflow as tf
from keras.src.optimizers import optimizer
import math
    

class Lamb(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        bias_correction=True,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        weight_decay=0.01,
        grad_averaging=True,
        max_grad_norm=1.0,
        trust_clip=False,
        always_adapt=False,
        caution=False,
        decoupled_decay=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="lamb",
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
        self.bias_correction = bias_correction
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_averaging = grad_averaging
        self.max_grad_norm = max_grad_norm
        self.trust_clip = trust_clip
        self.always_adapt = always_adapt
        self.caution = caution
        self.decoupled_decay = decoupled_decay
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.caution = False
        self.decoupled_decay = False
    
    def _get_clip_grad_norm(self):
        if self.max_grad_norm is None:
            return None

        norms = []
        for grad in self.grads:
            if tf.keras.backend.is_sparse(grad):
                raise RuntimeError("Lamb does not support sparse gradients, consider SparseAdam instead.")
            norms.append(tf.norm(grad))

        global_norm = tf.norm(tf.stack(norms))
        clip_global_norm = tf.clip_by_value(global_norm / self.max_grad_norm, clip_value_min=1.0, clip_value_max=float("inf"))
        return clip_global_norm

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
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.

        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.grads = grads
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        lr = learning_rate
        
        clip_grad_norm = self._get_clip_grad_norm() # None if disabled
        
        bias_correction = 1 if self.bias_correction else 0
        beta1 = self.beta1
        beta2 = self.beta2
        grad_averaging = 1 if self.grad_averaging else 0
        beta3 = 1 - beta1 if grad_averaging else 1.0
        
        # assume same step across group now to simplify things
        # per parameter step can be easily support by making it tensor, or pass list into kernel
        step = tf.get_static_value(self.iterations + 1)
        
        if bias_correction:
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
        else:
            bias_correction1, bias_correction2 = 1.0, 1.0
        
        for p, grad in zip(trainable_variables, grads):
            if clip_grad_norm is not None:
                grads[self._get_variable_index(p)] = grad / clip_grad_norm
            
            exp_avg = self.exp_avg[self._get_variable_index(p)]
            exp_avg_sq = self.exp_avg_sq[self._get_variable_index(p)]
    
            # Decay the first and second moment running average coefficient
            exp_avg.assign(beta1 * exp_avg + beta3 * grad)  # m_t
            exp_avg_sq.assign(beta2 * exp_avg_sq + (1 - beta2) * tf.square(grad))  # v_t
    
            denom = (tf.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + self.epsilon
            update = (exp_avg / bias_correction1) / denom
    
            if self.caution:
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                mask = tf.cast(update * grad > 0, dtype=grad.dtype)
                mask /= tf.maximum(tf.reduce_mean(mask), 1e-3)
                update *= mask
    
            if self.weight_decay != 0:
                if self.decoupled_decay:
                    p.assign_add(-lr * self.weight_decay * p)
                else:
                    update += self.weight_decay * p
    
            if self.weight_decay != 0 or self.always_adapt:
                # Layer-wise LR adaptation
                w_norm = tf.norm(p, ord=2)
                g_norm = tf.norm(update, ord=2)
                trust_ratio = w_norm / g_norm
                trust_ratio = tf.where(
                    w_norm > 0,
                    tf.where(g_norm > 0, trust_ratio, 1.0),
                    1.0,
                )
                if self.trust_clip:
                    trust_ratio = tf.minimum(trust_ratio, 1.0)
                update *= trust_ratio
    
            # Update parameters
            p.assign_add(-lr * update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bias_correction": self.bias_correction,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "grad_averaging": self.grad_averaging,
                "max_grad_norm": self.max_grad_norm,
                "trust_clip": self.trust_clip,
                "always_adapt": self.always_adapt,
                "caution": self.caution,
                "decoupled_decay": self.decoupled_decay,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass