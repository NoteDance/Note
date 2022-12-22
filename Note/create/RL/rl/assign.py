from tensorflow.python.ops import state_ops


def assign(param,new_param):
    for i in range(len(param)):
        state_ops.assign(param[i],new_param[i])
    return


def assign_pt(nn,grad):
    i=0
    for param in nn.parameters():
        param.grad=grad[i]
        i+=1
    return
