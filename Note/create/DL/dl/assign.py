def assign(nn,grad):
    i=0
    for param in nn.parameters():
        param.grad=grad[i]
        i+=1
    return