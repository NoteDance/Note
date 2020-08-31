import numpy as np
import pickle
import time


class policy_iteration:
    def __init__(self,policy,state,action,prs,discount=None,theta=None,end_flag=None):
        self.policy=policy
        self.state=state
        self.action=action
        self.prs=prs
        self.discount=discount
        self.theta=theta
        self.delta=0
        self.end_flag=end_flag
        self.iteration_num=0
        self.total_iteration_sum=0
        self.time=0
        self.total_time=0
        
        
    def policy_evaluation(self,policy,state,action,prs,discount,theta,iteration):
        if iteration==None:
            iteration=int(len(state)*3)
        V=np.zeros(len(state),dtype=np.float32)
        for i in range(iteration):
            delta=0
            for s in range(len(state)):
                v=0
                for a,action_prob in enumerate(policy[state[s]]):
                    for prob,reward,next_state,done in prs[state[s]][action[a]]:
                        v+=action_prob*prob*(reward+discount*V[next_state])
                delta=max(delta,np.abs(v-V[s]))
                V[s]=v
            if delta<=theta:
                break
        return V


    def policy_improvement(self,policy,V,state,action,prs,discount,flag,end_flag):
        for s in range(len(state)):
            old_action=np.argmax(policy[state[s]])
            action_value=np.zeros(len(action),dtype=np.float32)
            old_action_value=0
            for a in range(len(action)):
                for prob,reward,next_state,done in prs[state[s]][action[a]]:
                    action_value[a]+=prob*(reward+discount*V[next_state])
                    if done and next_state!=end_flag and end_flag!=None:
                        action_value[a]=float('-inf')
            best_action=np.max(action_value)
            for prob,reward,next_state,done in prs[state[s]][action[old_action]]:
                    old_action_value+=prob*(reward+discount*V[next_state])
            if old_action!=best_action and old_action_value!=best_action:
                flag=False
            policy[state[s]]=np.eye(len(action),dtype=np.float32)[best_action]
        return policy,flag


    def learn(self,iteration=None,path=None,one=True):
        self.delta=0
        t1=time.time()
        while True:
            flag=True
            V=self.policy_evaluation(self.policy,self.state,self.action,self.prs,self.discount,self.theta,iteration)
            self.policy,flag=self.policy_improvement(self.policy,V,self.state,self.action,self.prs,self.discount,flag,self.end_flag)
            if iteration%10!=0:
                temp=iteration-iteration%10
                temp=int(temp/10)
            else:
                temp=iteration/10
            if temp==0:
                temp=1
            if self.iteration_num%temp_iteration==0:
                if path!=None and self.iteration_num%iteration*2==0:
                    self.save(path,self.iteration_num,one)
            self.iteration_num+=1
            self.total_iteration_sum+=1
            if flag:
                t2=time.time()
                _time=(t2-t1)-int(t2-t1)
                if _time<0.5:
                    self.time=int(t2-t1)
                else:
                    self.time=int(t2-t1)+1
                self.total_time+=self.time
                print()
                print('time:{0}s'.format(self.time))
                return self.policy
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.discount)
        pickle.dump(self.theta)
        pickle.dump(self.end_flag)
        pickle.dump(self.total_iteration_sum)
        pickle.dump(self.total_time)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.end_flag=pickle.load(input_file)
        self.total_iteration_sum=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return