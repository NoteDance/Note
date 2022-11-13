def assign(nn,ac,grad=None):
    if grad==None:
        for param in nn.parameters():
            param.grad=ac*param.grad
    else:
        for param in nn.parameters():
            param.grad=ac*grad
    return
