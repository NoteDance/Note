def attenuate(attenuate,nn,t):
    if type(nn)==list:
        for i in range(len(nn)):
            for param in nn[i].parameters():
                param.grad=attenuate(t)*param
    else:
        for param in nn.parameters():
            param.grad=attenuate(t)*param
    return
