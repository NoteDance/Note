class Layers:
    def __init__(self):
        self.layer=[]
        self.param=[]
        self.saved_data=[]
        self.save_data_flag=[]
        self.use_data_flag=[]
        self.save_data_count=0
        self.output_size=None
        self.train_flag=True
    
    
    def add(self,layer,save_data=False,use_data=False):
        if type(layer)!=list:
            if save_data==True:
                self.save_data_count+=1
            if use_data==True and hasattr(layer,'save_data_count'):
                layer.save_data_count=self.save_data_count
            if use_data==True:
                self.save_data_count=0
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
            self.save_data_flag.append(save_data)
            self.use_data_flag.append(use_data)
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
                if self.use_data_flag[i]==False:
                    data=layer(data)
                else:
                    if hasattr(layer,'save_data_count'):
                        data=layer(self.saved_data)
                    else:
                        data=layer(data,self.saved_data.pop(0))
            else:
                if self.use_data_flag[i]==False:
                    data=layer(data,train_flag)
                else:
                    if hasattr(layer,'save_data_count'):
                        data=layer(self.saved_data,train_flag)
                    else:
                        data=layer(data,self.saved_data.pop(0),train_flag)
            if self.save_data_flag[i]==True:
                self.saved_data.append(data)
        return data
