import traceback


def check(nn,platform,data,labels):
    try:
        if platform.DType!=None:
            try:
                if nn.GradientTape!=None:
                    tape,output,loss=nn.GradientTape(data,labels)
            except Exception:
                print(traceback.format_exc())
                with platform.GradientTape(persistent=True) as tape:
                    try:
                        output=nn.fp(data)
                        loss=nn.loss(output,labels)
                    except Exception:
                        print(traceback.format_exc())
                        output,loss=nn.fp(data,labels)
            try:
                if nn.accuracy!=None:
                    nn.accuracy(output,labels)
            except Exception:
                print(traceback.format_exc())
                pass
            try:
                gradient=nn.gradient(tape,loss)
            except Exception:
                print(traceback.format_exc())
                gradient=tape.gradient(loss,nn.param)
            try:
                nn.opt.apply_gradients(zip(gradient,nn.param))
            except Exception:
                print(traceback.format_exc())
                nn.opt(gradient)
            print('No error')
            return
    except Exception:
        print(traceback.format_exc())
        output=nn.fp(data)
        loss=nn.loss(output,labels)
        try:
            if nn.accuracy!=None:
                nn.accuracy(output,labels)
        except Exception:
            print(traceback.format_exc())
            pass
        try:
            nn.opt.zero_grad()
            loss.backward()
            nn.opt.step()
        except Exception:
            print(traceback.format_exc())
            nn.opt(loss)
        print('No error')
        return
