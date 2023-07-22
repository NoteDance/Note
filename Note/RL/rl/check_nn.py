import numpy as np


def check(nn,platform,action_num=None):
    s=nn.env(initial=True)
    if hasattr(nn,'nn'):
        if hasattr(platform,'DType'):
            s=np.expand_dims(s,axis=0)
            a=np.argmax(nn.nn.fp(s))
        else:
            s=np.expand_dims(s,axis=0)
            s=platform.tensor(s,dtype=platform.float).to(nn.device)
            a=nn.nn(s).detach().numpy().argmax()
    else:
        if hasattr(nn,'action'):
            if hasattr(platform,'DType'):
                s=np.expand_dims(s,axis=0)
                a=nn.action(s).numpy()
            else:
                s=np.expand_dims(s,axis=0)
                s=platform.tensor(s,dtype=platform.float).to(nn.device)
                a=nn.action(s).detach().numpy()
        else:
            if hasattr(platform,'DType'):
                s=np.expand_dims(s,axis=0)
                a=nn.actor.fp(s).numpy()
                a=np.squeeze(a)
            else:
                s=np.expand_dims(s,axis=0)
                s=platform.tensor(s,dtype=platform.float).to(nn.device)
                a=nn.actor(s).detach().numpy()
                a=np.squeeze(a)
    next_s,r,done,_=nn.genv.step(a)
    if hasattr(nn,'pr'):
        nn.pr.TD=np.append(nn.pr.TD,nn.initial_TD)
    a=np.array(a)
    a=np.expand_dims(a,axis=0)
    next_s=np.expand_dims(next_s,axis=0)
    if hasattr(platform,'DType'):
        with platform.GradientTape(persistent=True) as tape:
            loss=nn.loss(s,a,next_s,r,done)
        if hasattr(nn,'gradient'):
            gradient=nn.gradient(tape,loss)
            if hasattr(nn.opt,'apply_gradients'):
                nn.opt.apply_gradients(zip(gradient,nn.param))
            else:
                nn.opt(gradient)
        else:
            if hasattr(nn,'nn'):
                gradient=tape.gradient(loss,nn.param)
                nn.opt.apply_gradients(zip(gradient,nn.param))
            else:
                actor_gradient=tape.gradient(loss[0],nn.param[0])
                critic_gradient=tape.gradient(loss[1],nn.param[1])
                nn.opt.apply_gradients(zip(actor_gradient,nn.param[0]))
                nn.opt.apply_gradients(zip(critic_gradient,nn.param[1]))
    else:
        loss=nn.loss(s,a,next_s,r,done)
        nn.backward(loss)
        nn.opt()
    print('No error')
    return
