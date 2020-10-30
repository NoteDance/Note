import numpy as np
import pickle
import time


class value_iteration:
    def __init__(self,policy,state_name,action_name,prs,discount,theta,end_flag=None):
        self.policy=policy
        self.state_name=state_name
        self.action_name=action_name
        self.prs=prs
        self.state_len=len(self.state_name)
        self.action_len=len(self.action_name)
        self.discount=discount
        self.theta=theta
        self.delta=0
        self.end_flag=end_flag
        self.iteration_num=0
        self.total_iteration_sum=0
        self.time=0
        self.total_time=0
    
    
    def init(self):
        self.V=np.zeros(len(self.state_name),dtype=np.float16)
        self.A=np.zeros(len(self.action_name),dtype=np.float16)
        return
    
    def learn(self,iteration=None,path=None,one=True):
        if iteration==None:
            iteration=int(len(self.state_name)*3)
        self.delta=0
        for i in range(iteration):
            t1=time.time()
            delta=0
            for s in range(len(self.state_name)):
                for a in range(len(self.action_name)):
                    for prob,r,next_s,done in self.prs[self.state_name[s]][self.action_name[a]]:
                        self.A[a]+=prob*(r+self.discount*self.V[next_s])
                        if done and next_s!=self.end_flag and self.end_flag!=None:
                            self.A[a]=float('-inf')
                            break
                best_action_value=max(self.A)
                delta=max(delta,np.abs(best_action_value-self.V[s]))
                self.V[s]=best_action_value
            if iteration%10!=0:
                temp=iteration-iteration%10
                temp=int(temp/10)
            else:
                temp=iteration/10
            if temp==0:
                temp=1
            if i%temp==0:
                print('iteration:{0}   delta:{1:.6f}'.format(i+1,delta))
                if path!=None and i%iteration*2==0:
                    self.save(path,i,one)
            self.iteration_num+=1
            self.total_iteration_sum+=1
            t2=time.time()
            self.time+=(t2-t1)
            if delta<=self.theta:
                self.time=self.time-int(self.time)
                if self.time<0.5:
                    self.time=int(self.time)
                else:
                    self.time=int(self.time)+1
                self.total_time+=self.time
                self.delta=delta
                print()
                print('last delta:{0:.6f}'.format(delta))
                print('time:{0}s'.format(self.time))
                break
        for s in range(len(self.state)):
            for s in range(len(self.state_name)):
                for a in range(len(self.action_name)):
                    for prob,r,next_s in self.prs[self.state_name[s]][self.action_name[a]]:
                        self.A[a]+=prob*(r+self.discount*self.V[next_s])
                best_a=np.argmax(self.A)
                self.policy[self.state_name[s]][best_a]=1
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.state_len,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.V,output_file)
        pickle.dump(self.A,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.theta,output_file)
        pickle.dump(self.delta,output_file)
        pickle.dump(self.end_flag,output_file)
        pickle.dump(self.total_iteration_sum,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.state_len=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        if self.state_len==len(self.state_name):
            self.V=pickle.load(input_file)
        if self.action_len==len(self.action_name):
            self.A=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.end_flag=pickle.load(input_file)
        self.total_iteration_sum=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
