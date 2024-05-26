class Layers:
    def __init__(self):
        self.layer=[]
        self.param=[]
        self.train_flag=True
    
    
    def add(self,layer):
        if type(layer)!=list:
            if hasattr(layer,'build'):
                if hasattr(layer,'input_size'):
                    if layer.input_size==None and self.output_size!=None:
                        layer.input_size=self.output_size
                        layer.build()
                        self.layer.append(layer)
                    else:
                        self.layer.append(layer)
                else:
                    self.layer.append(layer)
            else:
                self.layer.append(layer)
            if hasattr(layer,'param'):
                self.param.append(layer.param)
            if hasattr(layer,'output_size'):
                self.output_size=layer.output_size
        else:
            for layer in layer:
                if hasattr(layer,'build'):
                    if hasattr(layer,'input_size'):
                        if layer.input_size==None and self.output_size!=None:
                            layer.input_size=self.output_size
                            layer.build()
                            self.layer.append(layer)
                        else:
                            self.layer.append(layer)
                    else:
                        self.layer.append(layer)
                else:
                    self.layer.append(layer)
                if hasattr(layer,'param'):
                    self.param.append(layer.param)
                if hasattr(layer,'output_size'):
                    self.output_size=layer.output_size
        return
    
    
    def __call__(self,data,train_flag=True):
        for i,layer in enumerate(self.layer):
            if not hasattr(layer,'train_flag'):
                data=layer(data)
            else:
                data=layer(data,train_flag)
        return data
