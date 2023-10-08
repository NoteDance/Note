import tensorflow as tf


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
