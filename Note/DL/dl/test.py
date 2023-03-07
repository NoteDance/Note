import numpy as np


def test(nn,test_data,test_labels,platform,batch=None,loss=None,acc_flag='%'):
    if type(test_data)==list:
        data_batch=[x for x in range(len(test_data))]
    if type(test_labels)==list:
        labels_batch=[x for x in range(len(test_labels))]
    if batch!=None:
        total_loss=0
        total_acc=0
        if type(test_data)==list:
            batches=int((test_data[0].shape[0]-test_data[0].shape[0]%batch)/batch)
            shape0=test_data[0].shape[0]
        else:
            batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
            shape0=test_data.shape[0]
        for j in range(batches):
            index1=j*batch
            index2=(j+1)*batch
            if type(test_data)==list:
                for i in range(len(test_data)):
                    data_batch[i]=test_data[i][index1:index2]
            else:
                data_batch=test_data[index1:index2]
            if type(test_labels)==list:
                for i in range(len(test_labels)):
                    labels_batch[i]=test_labels[i][index1:index2]
            else:
                labels_batch=test_labels[index1:index2]
            try:
                output=nn.fp(data_batch)
            except AttributeError:
                output=nn(data_batch)
            if loss==None:
                batch_loss=nn.loss(output,labels_batch)
            else:
                batch_loss=loss(labels_batch,output)
            total_loss+=batch_loss
            try:
                if nn.accuracy!=None:
                    pass
                batch_acc=nn.accuracy(output,labels_batch)
                total_acc+=batch_acc
            except AttributeError:
                pass
        if shape0%batch!=0:
            batches+=1
            index1=batches*batch
            index2=batch-(shape0-batches*batch)
            try:
                if type(test_data)==list:
                    for i in range(len(test_data)):
                        data_batch[i]=platform.concat([test_data[i][index1:],test_data[i][:index2]],0)
                else:
                    data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)
                if type(test_labels)==list:
                    for i in range(len(test_labels)):
                        labels_batch[i]=platform.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                else:
                    labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)
            except:
                if type(test_data)==list:
                    for i in range(len(test_data)):
                        data_batch[i]=platform.concat([test_data[i][index1:],test_data[i][:index2]],0)
                else:
                    data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)
                if type(test_labels)==list:
                    for i in range(len(test_labels)):
                        labels_batch[i]=platform.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                else:
                    labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)
            try:
                output=nn.fp(data_batch)
            except AttributeError:
                output=nn(data_batch)
            if loss==None:
                batch_loss=nn.loss(output,labels_batch)
            else:
                batch_loss=loss(labels_batch,output)
            total_loss+=batch_loss
            try:
                if nn.accuracy!=None:
                    pass
                batch_acc=nn.accuracy(output,labels_batch)
                total_acc+=batch_acc
            except AttributeError:
                pass
        test_loss=total_loss.numpy()/batches
        test_loss=test_loss.astype(np.float32)
        try:
            if nn.accuracy!=None:
                test_acc=total_acc.numpy()/batches
                test_acc=test_acc.astype(np.float32)
        except AttributeError:
            pass
    else:
        try:
            output=nn.fp(test_data)
        except AttributeError:
            output=nn(test_data)
        if loss==None:
            test_loss=nn.loss(output,test_labels)
        else:
            test_loss=loss(test_labels,output)
        test_loss=test_loss.numpy().astype(np.float32)
        try:
            if nn.accuracy!=None:
                test_acc=nn.accuracy(output,test_labels)
                test_acc=test_acc.numpy().astype(np.float32)
        except AttributeError:
            pass
    print('test loss:{0:.6f}'.format(test_loss))
    try:
        if nn.accuracy!=None:
            pass
        if acc_flag=='%':
            print('accuracy:{0:.1f}'.format(test_acc*100))
        else:
            print('accuracy:{0:.6f}'.format(test_acc))
        if acc_flag=='%':
            return test_loss,test_acc*100
        else:
            return test_loss,test_acc
    except AttributeError:
        return test_loss
