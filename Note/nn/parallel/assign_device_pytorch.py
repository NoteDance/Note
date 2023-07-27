import torch


def assign_device(p, device_type): # a function to assign device according to the process index p and the device type
    if device_type == 'GPU': # if the device type is GPU
        if torch.cuda.is_available(): # if there are any available GPU devices
            try:
                torch.cuda.set_device(p % torch.cuda.device_count()) # set the device with index p modulo the number of devices as current
                device = torch.device('cuda', p % torch.cuda.device_count()) # create a torch.device object with the current device
            except RuntimeError as e: # catch any runtime error
                raise e # raise the error message
        else: # if there are no available GPU devices
            device = torch.device('cpu') # use CPU device as default
    elif device_type == 'CPU': # if the device type is CPU
        device = torch.device('cpu') # use CPU device as default
    else: # if the device type is neither GPU nor CPU
        raise ValueError('Invalid device type') # raise a value error
    return device
