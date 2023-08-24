class Layers:
    def __init__(self):
        self.layer=[]
        self.param=[]
    
    
    def add(self,layer):
        self.layer.append(layer)
        if hasattr(layer,'param'):
            self.param.append(layer.param)
        if hasattr(layer,'output_size'):
            self.output_size=layer.output_size
        return
    
    
    def output(self,data,train_flag=True):
        for layer in self.layer:
            if hasattr(layer,'output'):
                if not hasattr(layer,'train_flag'):
                    data=layer.output(data)
                else:
                    if not train_flag:
                        data=layer.output(data,train_flag)
                    else:
                        data=layer.output(data)
            else:
                data=layer(data)
        return data