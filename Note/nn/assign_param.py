from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest


def assign_param(self,param1,param2):
    parameter_flat1=nest.flatten(param1)
    parameter_flat2=nest.flatten(param2)
    for i in range(len(parameter_flat1)):
        state_ops.assign(parameter_flat1[i],parameter_flat2[i])
    return
