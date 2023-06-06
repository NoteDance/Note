import Note.nn.activation as a
import Note.nn.initializer as i


class dense:
    def __init__(self,weight_shape,weight_initializer='Xavier',bias_initializer='zero',activation=None,dtype='float32',use_bias=True):
        self.weight=i.initializer(weight_shape,weight_initializer,dtype)
        if use_bias==True:
            self.bias=i.initializer([weight_shape[1]],bias_initializer,dtype)
        self.activation=activation
        self.use_bias=use_bias
        if use_bias==True:
            self.weight_list=[self.weight,self.bias]
        else:
            self.weight_list=[self.weight]
    
    
    def output(self,data):
        return a.activation(data,self.weight,self.bias,self.activation,self.use_bias)
