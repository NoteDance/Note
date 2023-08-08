import numpy as np


def segment_data(data,labels,process):
    if len(data)!=process:
        length=len(data)-len(data)%process
        data=data[:length]
        labels=labels[:length]
        data=np.split(data,process)
        labels=np.split(labels,process)
        data=np.stack(data,axis=0)
        labels=np.stack(labels,axis=0)
        return data,labels
