def check(nn,platform,data,labels):
    try:
        if platform.DType!=None:
            try:
                if hasattr(nn,'GradientTape'):
                    tape,output,loss=nn.GradientTape(data,labels)
                else:
                    with platform.GradientTape(persistent=True) as tape:
                        output=nn.fp(data)
                        loss=nn.loss(output,labels)
            except Exception as e:
                raise e
            if hasattr(nn,'accuracy'):
                nn.accuracy(output,labels)
            if hasattr(nn,'gradient'):
                gradient=nn.gradient(tape,loss)
            else:
                gradient=tape.gradient(loss,nn.param)
            if hasattr(nn.opt,'apply_gradients'):
                nn.opt.apply_gradients(zip(gradient,nn.param))
            else:
                nn.opt(gradient)
            print('No error')
            return
    except Exception:
        output=nn.fp(data)
        loss=nn.loss(output,labels)
        if hasattr(nn,'accuracy'):
            nn.accuracy(output,labels)
        if hasattr(nn.opt,'zero_grad'):
            nn.opt.zero_grad()
            loss.backward()
            nn.opt.step()
        else:
            nn.opt(loss)
        print('No error')
        return
