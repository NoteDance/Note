import tensorflow as tf
import numpy as np
import pickle


class ART1:
    def __init__(self,data=None,r_neure=None,p=None):
        self.data=data
        self.c_neure=data.shape[-1]
        self.r_neure=r_neure
        self.p=p
    
    
    def init(self):
        self.W=tf.ones(shape=[self.c_neure,self.r_neure],dtype=tf.float16)/(self.c_neure+1)
        self.T=tf.ones(shape=[self.r_neure,self.c_neure],dtype=tf.int8)
        return
    
    
    def r_layer(self,data):
        data=data.reshape([1,-1])
        s=tf.matmul(data,self.W)
        return s,self.T[np.argmax(s)]
    
    
    def c_layer(self,t,data):
        return np.sum(t*data)/np.sum(data)
    
    
    def _search(self,s,t,data,vector):
        a=1
        resonance=False
        while True:
            s=s*vector
            t=self.T[np.argmax(s)]
            sim=self.c_layer(t,data)
            if sim>=self.p:
                resonance=True
            if resonance==True:
                self.W[:,np.argmax(s)]=t*data/(0.5+np.sum(t*data))
                self.T[np.argmax(s)]=t*data
                resonance=False
                self.accumulator+=1
                break
            elif a==self.r_neure:
                tf.concat([self.W,tf.ones(shape=[self.c_neure,1],dtype=tf.float16)/(self.c_neure+1)],axis=1)
                tf.concat([self.T,tf.ones(shape=[1,self.c_neure],dtype=tf.int8)],axis=0)
                self.W[:,np.argmax(s)]=t*data/(0.5+np.sum(t*data))
                self.T[np.argmax(s)]=t*data
                self.r_neure+=1
                self._vector=tf.concat([self._vector,tf.ones([1],dtype=tf.int8)],axis=0)
                self.accumulator+=1
                break
            else:
                vector[np.argmax(s)]=0
                a+=1
        return
    
    
    def recognition_layer(self,data,W,T):
        data=data.reshape([1,-1])
        s=tf.matmul(data,W)
        return s,T[np.argmax(s)]
    
    
    def compare_layer(self,t,data):
        return np.sum(t*data)/np.sum(data)
    
    
    def search(self,s,t,W,T,data,p,vector):
        a=1
        resonance=False
        while True:
            s=s*vector
            t=T[np.argmax(s)]
            sim=np.sum(t*data)/np.sum(data)
            if sim>=p:
                resonance=True
            if resonance==True:
                W[:,np.argmax(s)]=t*data/(0.5+np.sum(t*data))
                T[np.argmax(s)]=t*data
                break
            elif a==len(s):
                tf.concat([W,tf.ones(shape=[len(s),1],dtype=tf.float16)/(len(s)+1)],axis=1)
                tf.concat([T,tf.ones(shape=[1,len(data)],dtype=tf.int8)],axis=0)
                W[:,np.argmax(s)]=t*data/(0.5+np.sum(t*data))
                T[np.argmax(s)]=t*data
                break
            else:
                vector[np.argmax(s)]=0
                a+=1
        return
    
    
    def learn(self):
        resonance=False
        self._vector=tf.ones(shape=[self.r_neure],dtype=tf.int8)
        while True:
            self.accumulator=0
            if self.accumulator==len(self.data):
                break
            for i in range(len(self.data)):
                s,t=self.r_layer(self.data[i])
                sim=self.c_layer(t,self.data[i])
                if sim>=self.p:
                    resonance=True
                if resonance==True:
                    self.W[:,np.argmax(s)]=t*self.data[i]/(0.5+np.sum(t*self.data[i]))
                    self.T[np.argmax(s)]=t*self.data[i]
                    resonance=False
                    self.accumulator+=1
                else:
                    vector=self._vector
                    vector[np.argmax(s)]=0
                    self._search(s,t,self.data[i],vector)
        return
    
    
    def save(self,path):
        output_file=open(path+'\save.dat','wb')
        path=path+'\save.dat'
        index=path.rfind('\\')
        parameter_file=open(path.replace(path[index+1:],'parameter.dat'),'wb')
        pickle.dump([self.W,self.T],parameter_file)
        pickle.dump(self.c_neure,output_file)
        pickle.dump(self.r_neure,output_file)
        pickle.dump(self.p,output_file)
        return
    
    
    def restore(self,s_path,p_path):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        parameter=pickle.load(parameter_file)
        self.W=parameter[0]
        self.T=parameter[1]
        self.c_neure=pickle.load(input_file)
        self.r_neure=pickle.load(input_file)
        self.p=pickle.load(input_file)
        return
