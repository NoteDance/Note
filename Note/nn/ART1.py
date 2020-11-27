import tensorflow as tf
import numpy as np
import pickle


class ART1:
    def __init__(self,data=None,r_neure=None,p=None):
        self.data=data
        self.c_neure=data.shape[-1]
        self.r_neure=r_neure
        self.p=p
        self.accumulator=0
    
    
    def init(self):
        self.W=tf.ones(shape=[self.c_neure,self.r_neure],dtype=tf.float16)/(self.c_neure+1)
        self.T=tf.ones(shape=[self.r_neure,self.c_neure],dtype=tf.int8)
        return
    
    
    def r_layer(self,data):
        s=tf.matmul(data,self.W)
        return s,self.T[np.argmax(s)]
    
    
    def c_layer(self,t,data):
        return np.sum(t*data)/np.sum(data)
    
    
    def _search(self,s,t,data,vector):
        a=0
        resonance=False
        while True:
            a+=1
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
            elif a>self.r_neure:
                tf.concat(self.W,tf.ones(shape=[self.r_neure],dtype=tf.float16)/(self.c_neure+1))
                tf.concat(self.T,tf.ones(shape=[self.c_neure],dtype=tf.int8))
                self.W[:,np.argmax(s)]=t*data/(0.5+np.sum(t*data))
                self.T[np.argmax(s)]=t*data
                self.accumulator+=1
            else:
                vector[np.argmax(s)]=0
        return
    
    
    def search(self,s,t,W,T,data,p,vector):
        a=0
        resonance=False
        while True:
            a+=1
            s=s*vector
            t=T[np.argmax(s)]
            sim=np.sum(t*data)/np.sum(data)
            if sim>=p:
                resonance=True
            if resonance==True:
                W[:,np.argmax(s)]=t*data/(0.5+np.sum(t*data))
                T[np.argmax(s)]=t*data
                break
            elif a>len(s):
                tf.concat(W,tf.ones(shape=[len(s)],dtype=tf.float16)/(len(s)+1))
                tf.concat(T,tf.ones(shape=[len(data)],dtype=tf.int8))
                W[:,np.argmax(s)]=t*data/(0.5+np.sum(t*data))
                T[np.argmax(s)]=t*data
            else:
                vector[np.argmax(s)]=0
        return
    
    
    def learn(self):
        resonance=False
        _vector=tf.ones(shape=[self.r_neure],dtype=tf.int8)
        while True:
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
                    vector=_vector
                    vector[np.argmax(s)]=0
                    self._search(s,t,self.data[i],vector)
        return
    
    
    def save(self,nn_path):
        output_file=open(nn_path+'.dat','wb')
        pickle.dump(self.c_neure,output_file)
        pickle.dump(self.r_neure,output_file)
        pickle.dump(self.W,output_file)
        pickle.dump(self.T,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.accumulator,output_file)
        return
    
    
    def restore(self,nn_path):
        input_file=open(nn_path,'rb')
        self.c_neure=pickle.load(input_file)
        self.r_neure=pickle.load(input_file)
        self.W=pickle.load(input_file)
        self.T=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.accumulator=pickle.load(input_file)
        return
    
    
    def use(self,data):
        s,t=self.r_layer(data)
        sim=self.c_layer(t,data)
        vector=tf.ones(shape=[self.r_neure],dtype=tf.int8)
        if sim>=self.p:
            return np.argmax(s)
        else:
            vector[np.argmax(s)]=0
            self.search(s,t,self.W,self.T,data,self.p,vector)
        return
