from sys import getsizeof


def count_memory(param):
    param_memory=0
    for i in range(param):
        param_memory+=getsizeof(param[i])
    return 2*param_memory