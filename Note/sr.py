import pickle


def save(data,path):
    output_file=open(path,'wb')
    pickle.dump(data,output_file)
    output_file.close()
    return


def restore(path):
    input_file=open(path,'rb')
    data=pickle.load(input_file)
    input_file.close()
    return data
