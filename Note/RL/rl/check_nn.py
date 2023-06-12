import numpy as np


def check(nn,platform,action_num=None):
    s=nn.env(initial=True)
    try:
        if nn.nn!=None:
            s=np.expand_dims(s,axis=0)
            try:
                a=nn.action(s).numpy()
                try:
                    if nn.discriminator!=None:
                        s=np.squeeze(s)
                except Exception:
                    pass
            except Exception:
                a=np.random.choice(action_num)
            next_s,r,done=nn.env(a)
    except Exception as e:
        first_exception=e
        try:
            if nn.action!=None:
                raise first_exception
        except Exception as e:
            try:
                s=np.expand_dims(s,axis=0)
                a=(nn.actor(s)+nn.noise()).numpy()
                next_s,r,done=nn.env(a)
            except Exception:
                raise e
    try:
        nn.pr.TD=np.append(nn.pr.TD,nn.initial_TD)
    except Exception as e:
        try:
            if nn.pr!=None:
                raise e
        except Exception:
            pass
    a=np.array(a)
    a=np.expand_dims(a,axis=0)
    next_s=np.expand_dims(next_s,axis=0)
    try:
        if platform.DType!=None: 
            with platform.GradientTape(persistent=True) as tape:
                loss=nn.loss(s,a,next_s,r,done)
            try:
                gradient=nn.gradient(tape,loss)
                try:
                    nn.opt.apply_gradients(zip(gradient,nn.param))
                except Exception:
                    nn.opt(gradient)
            except Exception:
                try:
                    if nn.nn!=None:
                        gradient=tape.gradient(loss,nn.param)
                        nn.opt.apply_gradients(zip(gradient,nn.param))
                except Exception:
                    actor_gradient=tape.gradient(loss[0],nn.param[0])
                    critic_gradient=tape.gradient(loss[1],nn.param[1])
                    nn.opt.apply_gradients(zip(actor_gradient,nn.param[0]))
                    nn.opt.apply_gradients(zip(critic_gradient,nn.param[1]))
    except Exception as e:
        first_exception=e
        try:
            if platform.DType!=None: 
                raise first_exception
        except Exception as e:
            try:
                loss=nn.loss(s,a,next_s,r,done)
                nn.backward(loss)
                nn.opt()
            except Exception: 
                raise e
    print('No error')
    return
