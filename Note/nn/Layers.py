class Layers:
    def __init__(self):
        self.layer=[]
        self.param=[]
        self.saved_data=[]
        self.save_data_flag=[]
        self.use_data_flag=[]
        self.save_data_count=0
        self.output_size_list=[]
    
    
    def add(self,layer,save_data=False,use_data=False,axis=None):
        if save_data==True:
            self.save_data_count+=1
        if use_data==True and hasattr(layer,'save_data_count'):
            layer.save_data_count=self.save_data_count
        if use_data==True and hasattr(layer,'concat')!=True:
            self.output_size_list=[]
        if use_data==True:
            self.save_data_count=0
        if hasattr(layer,'build'):
            if layer.input_size==None and self.output_size!=None:
                layer.input_size=self.output_size
                layer.build()
                self.layer.append(layer)
            else:
                self.layer.append(layer)
        else:
            self.layer.append(layer)
        if hasattr(layer,'param'):
            self.param.append(layer.param)
        if hasattr(layer,'output_size'):
            self.output_size=layer.output_size
        if hasattr(layer,'concat'):
            if layer.axis==-1 or layer.axis==2:
                self.output_size=self.output_size_list.pop(0)
                for i in range(1,layer.save_data_count):
                    self.output_size+=self.output_size_list.pop(0)
        self.save_data_flag.append(save_data)
        self.use_data_flag.append(use_data)
        if save_data==True:
            self.output_size_list.append(self.output_size)
        return
    
    
    def output(self,data,train_flag=True):
        for i,layer in enumerate(self.layer):
            if hasattr(layer,'output'):
                if not hasattr(layer,'train_flag'):
                    if self.use_data_flag[i]==False:
                        data=layer.output(data)
                    else:
                        if hasattr(layer,'save_data_count'):
                            data=layer.output(self.saved_data)
                        else:
                            data=layer.output(data,self.saved_data.pop(0))
                else:
                    if self.use_data_flag[i]==False:
                        if not train_flag:
                            data=layer.output(data,train_flag)
                        else:
                            data=layer.output(data)
                    else:
                        if hasattr(layer,'save_data_count'):
                            if not train_flag:
                                data=layer.output(self.saved_data,train_flag)
                            else:
                                data=layer.output(self.saved_data)
                        else:
                            if not train_flag:
                                data=layer.output(data,self.saved_data.pop(0),train_flag)
                            else:
                                data=layer.output(data,self.saved_data.pop(0))
                if self.save_data_flag[i]==True:
                    self.saved_data.append(data)
            elif not hasattr(layer,'concat'):
                if self.use_data_flag[i]==False:
                    data=layer(data)
                else:
                    data=layer(data,self.saved_data.pop(0))
                if self.save_data_flag[i]==True:
                    self.saved_data.append(data)
            else:
                data=layer.concat(self.saved_data)
                if self.save_data_flag[i]==True:
                    self.saved_data.append(data)
        return data
