def test(nn,data,labels):
    try:
        output=nn.fp(data)
        nn.loss(output,labels)
    except TypeError:
        output,loss=nn.fp(data,labels)
    try:
        if nn.accuracy!=None:
            nn.accuracy(output,labels)
    except AttributeError:
        pass
    print('No error')
    return
