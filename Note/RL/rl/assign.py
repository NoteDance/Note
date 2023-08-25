from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest


def assign(param,new_param):
    param_flat=nest.flatten(param)
    new_param_flat=nest.flatten(new_param)
    for i in range(len(new_param_flat)):
        state_ops.assign(param_flat[i],new_param_flat[i])
    param=nest.pack_sequence_as(param,param_flat)
    new_param=nest.pack_sequence_as(new_param,new_param_flat)
    return
