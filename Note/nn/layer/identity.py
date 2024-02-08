class identity:
    def __init__(self,input_size=None):
        self.input_size=input_size
        if input_size!=None:
            self.output_size=input_size
    
    
    def __call__(self,data):
        return data