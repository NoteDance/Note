import tensorflow as tf
import torch


def assign_device(p, device_type): # a function to assign device according to the process index p and the device type
    devices = tf.config.list_physical_devices(device_type) # get a list of available devices of the given type
    if devices: # if there are any devices of the given type
        try:
            tf.config.set_visible_devices(devices[p % len(devices)], device_type) # set the device with index p modulo the number of devices as visible
            device = '/' + device_type + ':' + str(p % len(devices)) # store the device name as an attribute
        except RuntimeError as e: # catch any runtime error
            raise e # raise the error message
    else: # if there are no devices of the given type
        device = '/CPU:0' # use CPU device as default
    return device


def assign_tpu(p): # a function to assign TPU device according to the process index p
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='') # create a resolver to connect to the TPU cluster
    tf.config.experimental_connect_to_cluster(resolver) # connect to the cluster
    tf.tpu.experimental.initialize_tpu_system(resolver) # initialize the TPU system
    device = '/job:worker/replica:0/task:0/device:TPU:' + str(p) # store the device name as an attribute
    return device


def assign_device_pytorch(p, device_type): # a function to assign device according to the process index p and the device type
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
