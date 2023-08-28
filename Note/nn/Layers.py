class Layers:
    def __init__(self):
        self.layer=[]
        self.param=[]
        self.saved_data=None
        self.save_data_flag=[]
        self.use_data_flag=[]
    
    
    def add(self,layer,save_data=False,use_data=False):
        self.layer.append(layer)
        if hasattr(layer,'param'):
            self.param.append(layer.param)
        if hasattr(layer,'output_size'):
            self.output_size=layer.output_size
        self.save_data_flag.append(save_data)
        self.use_data_flag.append(use_data)
        return
    
    
    def output(self,data,train_flag=True):
        for i,layer in enumerate(self.layer):
            if hasattr(layer,'output'):
                if not hasattr(layer,'train_flag'):
                    data=layer.output(data)
                else:
                    if not train_flag:
                        data=layer.output(data,train_flag)
                    else:
                        data=layer.output(data)
                if self.save_data_flag[i]==True:
                    self.saved_data=data
            else:
                if self.use_data_flag[i]==False:
                    data=layer(data)
                else:
                    try:
                        data=layer(data+self.saved_data)
                    except TypeError:
                        data=layer(data,self.saved_data)
                if self.save_data_flag[i]==True:
                    self.saved_data=data
        return data
