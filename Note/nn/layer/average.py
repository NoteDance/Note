class average:
    def __init__(self):
        self.save_data_count=None
        
        
    def output(self,data):
        output=data[0]
        for i in range(1,self.save_data_count):
            output+=data[i]
        return output/self.save_data_count