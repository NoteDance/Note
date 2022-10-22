def attenuate(attenuate,nn,oc):
    if type(nn)==list:
        for i in range(len(nn)):
            for param in nn[i].parameters():
                param.grad=attenuate(oc)*param.grad
    else:
        for param in nn.parameters():
            param.grad=attenuate(oc)*param.grad
    return