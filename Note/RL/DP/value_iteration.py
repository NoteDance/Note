import numpy as np
import pickle
import time


class value_iteration:
    def __init__(self,policy,state,action,prs,discount,theta,end_flag=None):
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
        
        
    def learn(self,iteration=None,path=None,one=True):
        if iteration==None:
            iteration=int(len(self.state)*3)
        V=np.zeros(len(self.state),dtype=np.float32)
        self.delta=0        
        for i in range(iteration):
			t1=time.time()
            delta=0
            for s in range(len(self.state)):
                A=np.zeros(len(self.action),dtype=np.float32)
                for a in range(len(self.action)):
                    for prob,reward,next_state,done in self.prs[self.state[s]][self.action[a]]:
                        A[a]+=prob*(reward+self.discount*V[next_state])
                        if done and next_state!=self.end_flag and self.end_flag!=None:
                            A[a]=float('-inf')
                            break
                best_action_value=max(A)
                delta=max(delta,np.abs(best_action_value-V[s]))
                V[s]=best_action_value
            if iteration%10!=0:
                temp=iteration-iteration%10
                temp=int(temp/10)
            else:
                temp=iteration/10
            if temp==0:
                temp=1
            if i%temp==0:
                print('iteration:{0}   delta:{1:.6f}'.format(i,delta))
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
            for s in range(len(self.state)):
                A=np.zeros(len(self.action),dtype=np.float32)
                for a in range(len(self.action)):
                    for prob,reward,next_state in self.prs[self.state[s]][self.action[a]]:
                        A[a]+=prob*(reward+self.discount*V[next_state])
                best_action=max(A)
                self.policy[self.state[s]][best_action]=1
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.discount)
        pickle.dump(self.theta)
        pickle.dump(self.delta)
        pickle.dump(self.end_flag)
        pickle.dump(self.total_iteration_sum)
        pickle.dump(self.total_time)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.end_flag=pickle.load(input_file)
        self.total_iteration_sum=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
