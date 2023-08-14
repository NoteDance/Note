import numpy as np


def segment_data(data,labels,process):
    if len(data)!=process:
        data=np.array_split(data,process)
        labels=np.array_split(labels,process)
        return data,labels
