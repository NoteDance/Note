def attenuate(attenuate,nn,t):
    for i in range(len(nn)):
        for param in nn[i].parameters():
            param.grad=attenuate(t)*param
    return