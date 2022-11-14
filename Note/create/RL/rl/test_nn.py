import numpy as np


def test(nn,action_num=None):
    s=nn.env(initial=True)
    try:
        if nn.nn!=None:
            s=np.expand_dims(s,axis=0)
            try:
                if nn.action!=None:
                    a=nn.action(s)
                    try:
                        if nn.discriminator!=None:
                            s=np.squeeze(s)
                    except AttributeError:
                        pass
            except AttributeError:
                a=np.random.choice(action_num)
            next_s,r,done=nn.env(a)
    except AttributeError:
        s=np.expand_dims(s,axis=0)
        a=(nn.actor(s)+nn.noise()).numpy()
        next_s,r,done=nn.env(a)
    try:
        if nn.pr!=None:
            nn.pr.TD=np.append(nn.pr.TD,nn.initial_TD)
    except AttributeError: 
        pass
    loss=nn.loss(s,a,next_s,r,done)
    nn.opt(loss)
    print('No error')
    return
