import numpy as np
import traceback


def test(nn,platform,action_num=None):
    s=nn.env(initial=True)
    try:
        if nn.nn!=None:
            s=np.expand_dims(s,axis=0)
            try:
                if nn.action!=None:
                    a=nn.action(s).numpy()
                    try:
                        if nn.discriminator!=None:
                            s=np.squeeze(s)
                    except Exception:
                        print(traceback.format_exc())
                        pass
            except Exception:
                print(traceback.format_exc())
                a=np.random.choice(action_num)
            next_s,r,done=nn.env(a)
    except Exception:
        print(traceback.format_exc())
        s=np.expand_dims(s,axis=0)
        a=(nn.actor(s)+nn.noise()).numpy()
        next_s,r,done=nn.env(a)
    try:
        if nn.pr!=None:
            nn.pr.TD=np.append(nn.pr.TD,nn.initial_TD)
    except Exception:
        print(traceback.format_exc())
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
                    print(traceback.format_exc())
                    nn.opt(gradient)
            except Exception:
                print(traceback.format_exc())
                try:
                    if nn.nn!=None:
                        gradient=tape.gradient(loss,nn.param)
                        nn.opt.apply_gradients(zip(gradient,nn.param))
                except Exception:
                    print(traceback.format_exc())
                    actor_gradient=tape.gradient(loss[0],nn.param[0])
                    critic_gradient=tape.gradient(loss[1],nn.param[1])
                    nn.opt.apply_gradients(zip(actor_gradient,nn.param[0]))
                    nn.opt.apply_gradients(zip(critic_gradient,nn.param[1]))
    except Exception:
        print(traceback.format_exc())
        loss=nn.loss(s,a,next_s,r,done)
        nn.backward(loss)
        nn.opt()
    print('No error')
    return
