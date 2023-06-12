def check(nn,platform,data,labels):
    try:
        if platform.DType!=None:
            try:
                if nn.GradientTape!=None:
                    tape,output,loss=nn.GradientTape(data,labels)
            except Exception as e:
                first_exception=e
                try:
                    with platform.GradientTape(persistent=True) as tape:
                        output=nn.fp(data)
                        loss=nn.loss(output,labels)
                except Exception as e:
                    raise e
                    raise first_exception
            try:
                nn.accuracy(output,labels)
            except Exception as e:
                try:
                  if nn.accuracy!=None:
                      raise e
                except Exception: 
                    pass
            try:
                gradient=nn.gradient(tape,loss)
            except Exception as e:
                first_exception=e
                try:
                    gradient=tape.gradient(loss,nn.param)
                except Exception as e:
                    raise e
                    raise first_exception
            try:
                nn.opt.apply_gradients(zip(gradient,nn.param))
            except Exception as e:
                first_exception=e
                try:
                    nn.opt(gradient)
                except Exception as e:
                    raise e
                    raise first_exception
            print('No error')
            return
    except Exception:
        output=nn.fp(data)
        loss=nn.loss(output,labels)
        try:
            nn.accuracy(output,labels)
        except Exception as e:
            try:
              if nn.accuracy!=None:
                  raise e
            except Exception: 
                pass
        try:
            nn.opt.zero_grad()
            loss.backward()
            nn.opt.step()
        except Exception as e:
            first_exception=e
            try:
                nn.opt(loss)
            except Exception as e:
                raise e
                raise first_exception
        print('No error')
        return
