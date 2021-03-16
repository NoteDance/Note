import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class CMAC:
    def __init__(self,data=None,labels=None,form=None,memory=None,acc=None,test_data=None,test_labels=None):
        self.data=data
        self.labels=labels
        self.test_data=test_data
        self.test_labels=test_labels
        self.C=None
        self.form=form
        self.memory=memory
        self.rate=None
        self.acc=acc
        self.epoch=0
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.loss_list=[]
        self.acc_list=[]
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.time=0
    
    
    def set_up(self,C=None,rate=None,end_loss=None,end_acc=None,end_test_loss=None,end_test_acc=None):
        if C!=None:
            self.C=C
        if rate!=None:
            self.rate=rate
        if end_loss!=None:
            self.end_loss=end_loss
        if end_acc!=None:
            self.end_acc=end_acc
        if end_test_loss!=None:
            self.end_test_loss=end_test_loss
        if end_test_acc!=None:
            self.end_test_acc=end_test_acc
        return
    
    
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
    
    
    def _learn(self,i,epoch,path,one):
        t1=time.time()
        loss=0
        test_loss=0
        self.epoch+=1
        for j in range(len(self.data)):
            address=self.mapping(self.data[j])
            output=self.output(address)
            loss+=(output-self.labels[j])**2
            for k in range(self.C):
                self.memory[address[k]]=self.memory[address[k]]+self.rate*(output-self.labels[j])/self.C
        if self.test_data!=None:
            for j in range(len(self.data)):
                address=self.mapping(self.test_data[j])
                output=self.output(address)
                test_loss+=(output-self.test_labels[j])**2
                for k in range(self.C):
                    self.memory[address[k]]=self.memory[address[k]]+self.rate*(output-self.test_labels[j])/self.C
        loss=loss/len(self.data)
        self.loss_list.append(loss)
        if self.test_data!=None:
            test_loss=loss/len(self.data)
            self.test_loss_list.append(test_loss)
        if self.acc!=None:
            acc=self.acc(self.data)
            self.acc_list.append(acc)
        if self.test_data!=None:
            test_acc=self.acc(self.test_data)
            self.test_acc_list.append(test_acc)
        if epoch%10!=0:
            d=epoch-epoch%10
            d=int(d/10)
        else:
            d=epoch/10
        if d==0:
            d=1
        if i%d==0:
            print('epoch:{0}   loss:{1:.6f}'.format(i,loss))
            print('epoch:{0}   acc:{1:.6f}'.format(i,acc))
            if path!=None and i%epoch*2==0:
                self.save(path,i,one)
        t2=time.time()
        self.time+=(t2-t1)
        return loss,acc,test_loss,test_acc
    
    
    def learn(self,epoch,path=None,one=True):
        if epoch!=None:
            for i in range(epoch):
                loss,acc,test_loss,test_acc=self._learn(i,epoch,path,one)
        else:
            while True:
                loss,acc,test_loss,test_acc=self._learn(i,epoch,path,one)
        self.loss_list.append(loss)
        self.acc_list.append(acc)
        if self.test_data!=None:
            self.test_loss_list.append(test_loss)
            self.test_acc_list.append(test_acc)
        print()
        print('last loss:{0:.6f}'.format(loss))
        print('last acc:{0:.6f}'.format(acc))
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
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.acc_list)
        plt.title('train acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        print('acc:{0}'.format(self.acc_list[-1]))
        return
    
    
    def test_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.test_acc_list)
        plt.title('test acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('test loss:{0:.6f}'.format(self.test_loss_list[-1]))
        print('test acc:{0}'.format(self.test_acc_list[-1]))
        return
    
    
    def comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.train_loss_list,'b-',label='train loss')
        if self.test_data!=None:
            plt.plot(np.arange(self.epoch+1),self.test_loss_list,'r-',label='test loss')
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.train_acc_list,'b-',label='train acc')
        if self.test_data!=None:
            plt.plot(np.arange(self.epoch+1),self.test_acc_list,'r-',label='test acc')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        print('acc:{0}'.format(self.acc_list[-1]))
        if self.test_data!=None:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss_list[-1]))
            print('test acc:{0}'.format(self.test_acc_list[-1]))
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump([self.C,self.form,self.memory],parameter_file)
        parameter_file.close()
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
        pickle.dump(self.acc_list,output_file)
        pickle.dump(self.test_loss_list,output_file)
        pickle.dump(self.test_acc_list,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.time,output_file)
        output_file.close()
        parameter_file.close()
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
        self.acc_list=pickle.load(input_file)
        self.test_loss_list=pickle.load(input_file)
        self.test_acc_list=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        input_file.close()
        parameter_file.close()
        return
