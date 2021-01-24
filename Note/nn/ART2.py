import tensorflow as tf
import numpy as np
import pickle


class ART2:
    def __init__(self,data=None,r_neure=None,p=None,epislon=None,a=10,b=10,c=0.1,d=0.9,e=0,theta=0.2):
        self.data=data
        self.c_neure=data.shape[-1]
        self.r_neure=r_neure
        self.p=p
        self.epislon=epislon
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.e=e
        self.theta=theta
    
    
    def init(self):
        self.W=tf.ones(shape=[self.c_neure,self.r_neure],dtype=tf.float16)/(1+self.d)*np.sqrt(self.c_neure)
        self.T=tf.zeros(shape=[self.r_neure,self.c_neure],dtype=tf.int8)
        return
    
    
    def f(self,x):
        if x>=self.theta:
            pass
        else:
            return (2*self.theta*x**2)/(x**2+self.theta**2)
    
    
    def r_layer(self,p):
        h=tf.matmul(p,self.W)
        return h,self.T[np.argmax(h)]
    
    
    def c_layer(self,data):
        w=0
        x=0
        v=0
        u=0
        p=0
        q=0
        data=data.reshape([1,-1])
        while True:
            u_=u
            w=data+self.a*u
            x=w/(self.e+np.sqrt(np.sum(w**2)))
            f1=self.f(x)
            f2=self.f(q)
            v=f1+self.b*f2
            u=v/(self.e+np.sqrt(np.sum(v**2)))
            p=u
            q=p/(self.e+np.sqrt(np.sum(p**2)))
            if np.sum(np.abs(u-u_)<=self.epislon)==len(u):
                break
        return u,p
    
    
    def reset(self,u,h,t):
        vector=self.r_vector
        vector[np.argmax(h)]=self.d
        p=u+vector*t
        r=(u+self.c*p)/(self.e+np.sqrt(np.sum(u**2))+np.sqrt(np.sum(p**2)))
        return np.sqrt(np.sum(r**2))
    
    
    def _search(self,u,h,t,vector):
        a=1
        resonance=False
        while True:
            h=h*vector
            t=self.T[np.argmax(h)]
            r=self.reset(u,h,t)
            if r>=self.p:
                resonance=True
            if resonance==True:
                self.W[:,np.argmax(h)]=self.W[:,np.argmax(h)]+self.d*(1-self.d)*(u/(1-self.d)-self.W[:,np.argmax(h)])
                self.T[np.argmax(h)]=self.T[np.argmax(h)]+self.d*(1-self.d)*(u/(1-self.d)-self.T[np.argmax(h)])
                resonance=False
                self.accumulator+=1
                break
            elif a==self.r_neure:
                tf.concat([self.W,tf.ones(shape=[self.c_neure,1],dtype=tf.float16)/(1+self.d)*np.sqrt(self.c_neure)],axis=1)
                tf.concat([self.T,tf.ones(shape=[1,self.c_neure],dtype=tf.int8)],axis=0)
                self.W[:,np.argmax(h)]=self.W[:,np.argmax(h)]+self.d*(1-self.d)*(u/(1-self.d)-self.W[:,np.argmax(h)])
                self.T[np.argmax(h)]=self.T[np.argmax(h)]+self.d*(1-self.d)*(u/(1-self.d)-self.T[np.argmax(h)])
                self.r_neure+=1
                self._vector=tf.concat([self._vector,tf.ones([1],dtype=tf.int8)],axis=0)
                self.accumulator+=1
                break
            else:
                vector[np.argmax(h)]=0
                a+=1
        return
    
    
    def recognition_layer(self,p,W,T):
        h=tf.matmul(p,W)
        return h,T[np.argmax(h)]
    
    
    def compare_layer(self,data,a,b,e,theta,epislon):
        w=0
        x=0
        v=0
        u=0
        p=0
        q=0
        data=data.reshape([1,-1])
        while True:
            u_=u
            w=data+a*u
            x=w/(e+np.sqrt(np.sum(w**2)))
            if x>=theta:
                pass
            else:
                f1=(2*theta*x**2)/(x**2+theta**2)
            if q>=theta:
                pass
            else:
                f2=(2*theta*q**2)/(q**2+theta**2)  
            v=f1+b*f2
            u=v/(e+np.sqrt(np.sum(v**2)))
            p=u
            q=p/(e+np.sqrt(np.sum(p**2)))
            if np.sum(np.abs(u-u_)<=epislon)==len(u):
                break
        return u,p
    
    
    def search(self,u,h,t,W,T,p,d,vector):
        a=1
        resonance=False
        while True:
            h=h*vector
            t=T[np.argmax(h)]
            r=self.reset(u,h,t)
            if r>=p:
                resonance=True
            if resonance==True:
                W[:,np.argmax(h)]=W[:,np.argmax(h)]+d*(1-d)*(u/(1-d)-W[:,np.argmax(h)])
                T[np.argmax(h)]=T[np.argmax(h)]+d*(1-d)*(u/(1-d)-T[np.argmax(h)])
                break
            elif a==len(h):
                tf.concat([W,tf.ones(shape=[len(h),1],dtype=tf.float16)/(1+d)*np.sqrt(len(h))],axis=1)
                tf.concat([T,tf.ones(shape=[1,len(t)],dtype=tf.int8)],axis=0)
                W[:,np.argmax(h)]=W[:,np.argmax(h)]+d*(1-d)*(u/(1-d)-W[:,np.argmax(h)])
                T[np.argmax(h)]=T[np.argmax(h)]+d*(1-d)*(u/(1-d)-T[np.argmax(h)])
                break
            else:
                vector[np.argmax(h)]=0
                a+=1
        return
    
    
    def learn(self):
        resonance=False
        self._vector=tf.ones(shape=[self.r_neure],dtype=tf.int8)
        self.r_vector=tf.zeros(shape=[self.r_neure],dtype=tf.int8)
        while True:
            self.accumulator=0
            if self.accumulator==len(self.data):
                break
            for i in range(len(self.data)):
                u,p=self.c_layer(self.data[i])
                h,t=self.r_layer(p)
                r=self.reset(u,h,t)
                if r>=self.p:
                    resonance=True
                if resonance==True:
                    self.W[:,np.argmax(h)]=self.W[:,np.argmax(h)]+self.d*(1-self.d)*(u/(1-self.d)-self.W[:,np.argmax(h)])
                    self.T[np.argmax(h)]=self.T[np.argmax(h)]+self.d*(1-self.d)*(u/(1-self.d)-self.T[np.argmax(h)])
                    resonance=False
                    self.accumulator+=1
                else:
                    vector=self._vector
                    vector[np.argmax(h)]=0
                    self._search(u,h,t,vector)
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump([self.W,self.T],parameter_file)
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
        pickle.dump(self.epislon,output_file)
        pickle.dump(self.a,output_file)
        pickle.dump(self.b,output_file)
        pickle.dump(self.c,output_file)
        pickle.dump(self.d,output_file)
        pickle.dump(self.e,output_file)
        pickle.dump(self.theta,output_file)
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
        self.epislon=pickle.load(input_file)
        self.a=pickle.load(input_file)
        self.b=pickle.load(input_file)
        self.c=pickle.load(input_file)
        self.d=pickle.load(input_file)
        self.e=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        return
