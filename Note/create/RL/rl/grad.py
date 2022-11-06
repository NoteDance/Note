def grad(nn):
    grad=[]
    for param in nn.parameters():
        grad.append(param.grad)
    return grad