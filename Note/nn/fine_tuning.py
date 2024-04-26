from Note.nn.layer.dense import dense

def fine_tuning(param,param_,layer,layer_,classes,dim,flag=0):
    _param=[]
    if flag==0:
        param_=param.copy()
        layer_=layer
        layer=dense(classes, dim)
        param.extend(layer.param)
        param=_param
    elif flag==1:
        del param_[-len(layer.param):]
        param_.extend(layer.param)
        param=param_
    else:
        layer,layer_=layer_,layer
        del param_[-len(layer.param):]
        param_.extend(layer.param)
        param=param_
    return