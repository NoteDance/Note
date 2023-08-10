import numpy as np


def segment_data(data,labels,process):
    if len(data)!=process:
        length=len(data)-len(data)%process
        data=data[:length]
        labels=labels[:length]
        data=np.split(data,process)
        labels=np.split(labels,process)
        return data,labels
