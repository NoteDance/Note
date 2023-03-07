import numpy as np


def segment_data(data,labels,pt_count):
    if len(data)!=pt_count:
        data_=None
        labels_=None
        segments=int((len(data)-len(data)%pt_count)/pt_count)
        for i in range(pt_count):
            index1=i*segments
            index2=(i+1)*segments
            if i==0:
                data_=np.expand_dims(data[index1:index2],axis=0)
                labels_=np.expand_dims(labels[index1:index2],axis=0)
            else:
                data_=np.concatenate((data_,np.expand_dims(data[index1:index2],axis=0)))
                labels_=np.concatenate((labels_,np.expand_dims(labels[index1:index2],axis=0)))
        if len(data)%pt_count!=0:
            segments+=1
            index1=segments*pt_count
            index2=pt_count-(len(data)-segments*pt_count)
            data_=np.concatenate((data_,np.expand_dims(data[index1:index2],axis=0)))
            labels_=np.concatenate((labels_,np.expand_dims(labels[index1:index2],axis=0)))
        return data_,labels_
