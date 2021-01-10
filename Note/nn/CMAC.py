import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class CMAC:
    def __init__(self,data=None,labels=None,C=None,form=None,memory=None,rate=None):
        self.data=data
        self.labels=labels
        self.C=C
        self.form=form
        self.memory=memory
        self.rate=rate
        self.epoch=0
        self.loss_list=[]
        self.time=0
    
    
    def mapping(self,data):
        _address=self.form[data]
        address=_address[0]
        for i in range(len(_address)-1):
            address=np.char.add(address,_address[i+1])
        return address
    
    
    def output(self,address):
        output=0
        for i in range(self.C):
            if address[i] not in self.memory:
                self.memory[address[i]]=0
            output+=self.memory[address[i]]
        return output
    
    
    def learn(self,epoch,path=None,one=True):
        for i in range(epoch):
            t1=time.time()
            loss=0
            self.epoch+=1
            for j in range(len(self.data)):
                address=self.mapping(self.data[j])
                output=self.output(address)
                loss+=(output-self.labels[j])**2
                for k in range(self.C):
                    self.memory[address[k]]=self.memory[address[k]]+self.rate*(output-self.labels[j])/self.C
            loss=loss/len(self.data)
            self.loss_list.append(loss)
            if epoch%10!=0:
                d=epoch-epoch%10
                d=int(d/10)
            else:
                d=epoch/10
            if d==0:
                d=1
            if i%d==0:
                print('epoch:{0}   loss:{1:.6f}'.format(i,loss))
                if path!=None and i%epoch*2==0:
                    self.save(path,i,one)
            t2=time.time()
            self.time+=(t2-t1)
        self.loss_list.append(loss)
        print()
        print('last loss:{0:.6f}'.format(loss))
        if self.time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        print('time:{0}s'.format(self.time))
        return
    
    
    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'\save.dat','wb')
            path=path+'\save.dat'
            index=path.rfind('\\')
            parameter_file=open(path.replace(path[index+1:],'parameter.dat'),'wb')
        else:
            output_file=open(path+'\save-{0}.dat'.format(i+1),'wb')
            path=path+'\save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            parameter_file=open(path.replace(path[index+1:],'parameter-{0}.dat'.format(i+1)),'wb')
        pickle.dump([self.C,self.form,self.memory],parameter_file)
        pickle.dump(self.rate,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.time,output_file)
        return
    
    
    def restore(self,s_path,p_path):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        parameter=pickle.load(parameter_file)
        self.C=parameter[0]
        self.form=parameter[1]
        self.memory=parameter[2]
        self.rate=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        return
