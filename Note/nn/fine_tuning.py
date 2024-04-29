from Note.nn.layer.dense import dense

def fine_tuning(param,param_,layer,layer_,classes,dim,flag):
    _param=[]
    if flag==0:
        param_[0]=param.copy()
        layer_[0]=layer
        layer=dense(classes,dim)
        param.extend(layer.param)
        param=_param
    elif flag==1:
        del param_[0][-len(layer.param):]
        param_[0].extend(layer.param)
        param=param_[0]
    else:
        layer,layer_[0]=layer_[0],layer
        del param_[0][-len(layer.param):]
        param_[0].extend(layer.param)
        param=param_[0]
    return
