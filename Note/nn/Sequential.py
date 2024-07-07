class Sequential:
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
            self.layer.append(layer)
            if hasattr(layer,'param'):
                self.param.extend(layer.param)
            if hasattr(layer,'output_size'):
                self.output_size=layer.output_size
            self.save_data_flag.append(save_data)
            self.use_data_flag.append(use_data)
        else:
            for layer in layer:
                self.layer.append(layer)
                if hasattr(layer,'param'):
                    self.param.extend(layer.param)
                if hasattr(layer,'output_size'):
                    self.output_size=layer.output_size
        return
    
    
    def __call__(self,data,training=True):
        for i,layer in enumerate(self.layer):
            if not hasattr(layer,'train_flag'):
                if len(self.use_data_flag)==0 or self.use_data_flag[i]==False:
                    data=layer(data)
                else:
                    if hasattr(layer,'save_data_count'):
                        data=layer(self.saved_data)
                    else:
                        data=layer(data,self.saved_data.pop(0))
            else:
                if len(self.use_data_flag)==0 or self.use_data_flag[i]==False:
                    data=layer(data,training)
                else:
                    if hasattr(layer,'save_data_count'):
                        data=layer(self.saved_data,training)
                    else:
                        data=layer(data,self.saved_data.pop(0),training)
            if len(self.save_data_flag)>0 and self.save_data_flag[i]==True:
                self.saved_data.append(data)
        return data
