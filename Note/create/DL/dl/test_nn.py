def test(nn,platform,data,labels):
    try:
        if platform.DType!=None:
            try:
                if nn.GradientTape!=None:
                    tape,output,loss=nn.GradientTape(data,labels)
            except AttributeError:
                with platform.GradientTape(persistent=True) as tape:
                    try:
                        output=nn.fp(data)
                        loss=nn.loss(output,labels)
                    except TypeError:
                        output,loss=nn.fp(data,labels)
            try:
                if nn.accuracy!=None:
                    nn.accuracy(output,labels)
            except AttributeError:
                pass
            try:
                gradient=nn.gradient(tape,loss)
            except AttributeError:
                gradient=tape.gradient(loss,nn.param)
            try:
                nn.opt.apply_gradients(zip(gradient,nn.param))
            except AttributeError:
                nn.opt(gradient)
            print('No error')
            return
    except AttributeError:
        output=nn.fp(data)
        loss=nn.loss(output,labels)
        try:
            if nn.accuracy!=None:
                nn.accuracy(output,labels)
        except AttributeError:
            pass
        try:
            nn.opt.zero_grad()
            loss.backward()
            nn.opt.step()
        except:
            nn.opt(loss)
        print('No error')
        return
