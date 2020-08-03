import numpy as np


class tf2:
    def __init__(self):
        self.batch=None
        self.batches=None
        self.index1=None
        self.index2=None
        
        
    def batch(self,data):
        if self.index1==self.batches*self.batch:
            return np.concatenate([data[self.index1:],data[:self.index2]])
        else:
            return data[self.index1:self.index2]
        
        
    def extend(self,variable):
        for i in range(len(variable)-1):
            variable[0].extend(variable[i+1])
        return variable[0]
        
        
    def apply_gradient(self,tape,optimizer,loss,variable):
        gradient=tape.gradient(loss,variable)
        optimizer.apply_gradients(zip(gradient,variable))
        return