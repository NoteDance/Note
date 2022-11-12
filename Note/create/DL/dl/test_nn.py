def test_nn(nn,platform,data,labels):
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
                if nn.opt!=None:
                    gradient=tape.gradient(loss,nn.param)
                    nn.opt.apply_gradients(zip(gradient,nn.param))
            except AttributeError:
                gradient=nn.gradient(tape,loss,nn.param)
                nn.oopt(gradient,nn.param)
            print('No error')
            return
    except AttributeError:
        output=nn.fp(data)
        loss=nn.loss(output,labels)
        try:
            nn.opt.zero_grad()
            loss.backward()
            nn.opt.step()
        except:
            nn.opt(loss)
        print('No error')
        return
