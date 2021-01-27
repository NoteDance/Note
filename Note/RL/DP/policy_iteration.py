import numpy as np
import pickle
import time


class policy_iteration:
    def __init__(self,policy,state_name,action_name,prs,discount=None,theta=None,end_flag=None):
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
        self.total_iteration=0
        self.time=0
        self.total_time=0
    
    
    def init_v(self):
        self._V=np.zeros(len(self.state_name),dtype=np.float16)
        self.action_value=np.zeros(len(self.action_name),dtype=np.float16)
        return
    
    
    def init(self,discount=None,theta=None,flag=None):
        if discount!=None:
            self.discount=discount
        if theta!=None:
            self.theta=theta
        if flag==None:
            self.delta=0
            self.iteration_num=0
            self.total_iteration=0
            self.time=0
            self.total_time=0
        return
    
    
    def policy_evaluation(self,policy,V,state_name,action_name,prs,discount,theta,iteration):
        if iteration==None:
            iteration=int(len(state_name)*3)
        for i in range(iteration):
            delta=0
            for s in range(len(state_name)):
                v=0
                for a,action_prob in enumerate(policy[state_name[s]]):
                    for prob,r,next_s,done in prs[state_name[s]][action_name[a]]:
                        v+=action_prob[a]*prob*(r+discount*V[next_s])
                delta=max(delta,np.abs(v-V[s]))
                V[s]=v
            if delta<=theta:
                break
        return V

    
    def policy_improvement(self,policy,action_value,V,state_name,action_name,prs,discount,flag,end_flag):
        for s in range(len(state_name)):
            old_a=np.argmax(policy[state_name[s]])
            old_action_value=0
            for a in range(len(action_name)):
                for prob,r,next_s,done in prs[state_name[s]][action_name[a]]:
                    action_value[a]+=prob*(r+discount*V[next_s])
                    if done and next_s!=end_flag and end_flag!=None:
                        action_value[a]=float('-inf')
            best_a=np.argmax(action_value)
            best_action_value=np.max(action_value)
            for prob,r,next_s,done in prs[state_name[s]][action_name[old_a]]:
                    old_action_value+=prob*(r+discount*V[next_s])
            if old_a!=best_a and old_action_value!=best_action_value:
                flag=False
            policy[state_name[s]]=np.eye(len(action_name),dtype=np.int8)[best_a]
        return policy,flag


    def learn(self,iteration=None,path=None,one=True):
        self.delta=0
        while True:
            t1=time.time()
            flag=True
            V=self.policy_evaluation(self.policy,self._V,self.state,self.action,self.prs,self.discount,self.theta,iteration)
            self.policy,flag=self.policy_improvement(self.policy,self.action_value,V,self.state,self.action,self.prs,self.discount,flag,self.end_flag)
            if iteration%10!=0:
                d=iteration-iteration%10
                d=int(d/10)
            else:
                d=iteration/10
            if d==0:
                d=1
            if self.iteration_num%d==0:
                if path!=None and self.iteration_num%iteration*2==0:
                    self.save(path,self.iteration_num,one)
            self.iteration_num+=1
            self.total_iteration+=1
            t2=time.time()
            self.time+=(t2-t1)
            if flag:
                self.time=self.time-int(self.time)
                if self.time<0.5:
                    self.time=int(self.time)
                else:
                    self.time=int(self.time)+1
                self.total_time+=self.time
                print()
                print('time:{0}s'.format(self.time))
                return
    
    
    def save_policy(self,path):
        output_file=open(path+'.dat','wb')
        pickle.dump(self.policy,output_file)
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.state_len,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self._V,output_file)
        pickle.dump(self.action_value,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.theta,output_file)
        pickle.dump(self.end_flag,output_file)
        pickle.dump(self.iteration_num,output_file)
        pickle.dump(self.total_iteration,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.state_len=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        self._V=pickle.load(input_file)
        self.action_value=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.end_flag=pickle.load(input_file)
        self.iteration_num=pickle.load(input_file)
        self.total_iteration=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
