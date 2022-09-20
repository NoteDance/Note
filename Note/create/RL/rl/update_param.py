def update_param(target_net,net,tau=None):
    if tau==None:
        target_net.load_state_dict(net.state_dict())
    else:
        for target_param,param in zip(target_net[0].parameters(),net[0].parameters()):
            target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
        for target_param,param in zip(target_net[1].parameters(),net[1].parameters()):
            target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
    return