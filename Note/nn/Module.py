class Module:
    param=[]
    ctl_list=[]
    ctsl_list=[]
    
    
    def convert_to_list():
        for ctl in Module.ctl_list:
            ctl()
        return
    
    
    def convert_to_shared_list(manager):
        for ctsl in Module.ctsl_list:
            ctsl(manager)
        return
    
    
    def init():
        Module.param.clear()
        Module.ctl_list.clear()
        Module.ctsl_list.clear()
        return
