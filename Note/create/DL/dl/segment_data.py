import numpy as np


def segment_data(data,labels,thread):
    if len(data)!=thread:
        data_=None
        labels_=None
        segments=int((len(data)-len(data)%thread)/thread)
        for i in range(thread):
            index1=i*segments
            index2=(i+1)*segments
            if i==0:
                data_=np.expand_dims(data[index1:index2],axis=0)
                labels_=np.expand_dims(labels[index1:index2],axis=0)
            else:
                data_=np.concatenate((data_,np.expand_dims(data[index1:index2],axis=0)))
                labels_=np.concatenate((labels_,np.expand_dims(labels[index1:index2],axis=0)))
        if len(data)%thread!=0:
            segments+=1
            index1=segments*thread
            index2=thread-(len(data)-segments*thread)
            data_=np.concatenate((data_,np.expand_dims(data[index1:index2],axis=0)))
            labels_=np.concatenate((labels_,np.expand_dims(labels[index1:index2],axis=0)))
        return data_,labels_