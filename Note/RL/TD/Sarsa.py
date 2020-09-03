import tensorflow as tf
import numpy as np
import pickle
import time


class Sarsa:
    def __init__(self,state,state_list,action,search_space,q=None,epsilon=None,alpha=None,discount=None,theta=None):
        self.q=q
        self.state=state
        self.state_list=state_list
        self.action=action
        self.search_space=search_space
        self.epsilon=epsilon
        self.alpha=alpha
        self.discount=discount
        self.theta=theta
        self.delta=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0


    def epsilon_greedy_policy(self,q,state,action):
        action_prob=np.ones(len(action),dtype=np.float32)
        action_prob=action_prob*self.epsilon/len(action)
        best_action=np.argmax(q[state])
        action_prob[best_action]+=1-self.epsilon
        return action_prob
    
    
    def td(self,q,reward,state,next_state,action):
        action_prob=self.epsilon_greedy_policy(q,self.state[next_state],self.action)
        next_action=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
        q[state][action]=q[state][action]+self.alpha*(reward+self.discount*q[next_state][next_action]-q[state][action])
        return q,next_state
    
    
    def update_q(self,q,state):
        a=0
        while True:
            action_prob=self.epsilon_greedy_policy(q,self.state[state],self.action)
            action=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
            next_state,reward,end=self.search_space[self.action[action]]
            temp=q[state][action]
            self.delta+=np.abs(q[state][action]-temp)
            if end:
                self.delta+=self.delta/a
                break
            q,next_state=self.td(q,reward,state,next_state,action)
            state=next_state
            a+=1
        return q
    
    
    def learn(self,episode_num,path=None,one=True):
        self.delta=0
        if self.q==None:
            self.q=np.zeros([len(self.state_list),len(self.action)],dtype=np.float32)
        elif len(self.state_list)>self.q.shape[0] or len(self.action)>self.q.shape[1]:
            q=self.q*tf.ones([len(self.state_list),len(self.action)],dtype=tf.float32)[:self.q.shape[0],:self.q.shape[1]]
            self.q=q.numpy()
        t1=time.time()
        for i in range(episode_num):
            s=np.random.choice(np.arange(len(self.state_list)),p=np.ones(len(self.state_list))*1/len(self.state_list))
            self.q=self.update_q(self.q,self.state_list[s])
            self.delta=self.delta/(i+1)
            if episode_num%10!=0:
                temp=episode_num-episode_num%10
                temp=int(temp/10)
            else:
                temp=episode_num/10
            if temp==0:
                temp=1
            if i%temp==0:
                print('episode_num:{0}   delta:{1:.6f}'.format(i,self.delta))
                if path!=None and i%episode_num*2==0:
                    self.save(path,i,one)
            self.episode_num+=1
            self.total_episode+=1
            if self.theta!=None and self.delta<=self.theta:
                break
        t2=time.time()
        _time=(t2-t1)-int(t2-t1)
        if _time<0.5:
            self.time=int(t2-t1)
        else:
            self.time=int(t2-t1)+1
        self.total_time+=self.time
        print()
        print('last delta:{0:.6f}'.format(self.delta))
        print('time:{0}s'.format(self.time))
        return self.q
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.epsilon)
        pickle.dump(self.alpha)
        pickle.dump(self.discount)
        pickle.dump(self.theta)
        pickle.dump(self.delta)
        pickle.dump(self.total_episode)
        pickle.dump(self.total_time)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.epsilon=pickle.load(input_file)
        self.alpha=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return