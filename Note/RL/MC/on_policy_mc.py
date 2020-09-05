import tensorflow as tf
import numpy as np
import pickle
import time


class on_policy_mc:
    def __init__(self,q,state,state_list,action,search_space,epsilon=None,discount=None,theta=None,episode_step=None,save_episode=True):
        self.q=q
        self.episode=[]
        self.r_sum=dict()
        self.r_count=dict()
        self.state=state
        self.state_list=state_list
        self.action=action
        self.search_space=search_space
        self.epsilon=epsilon
        self.discount=discount
        self.theta=theta
        self.episode_step=episode_step
        self.save_episode=save_episode
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
    
    
    def episode(self,q,state,action,search_space):
        episode=[]
        _episode=[]
        if self.episode_step==None:
            while True:
                action_prob=self.epsilon_greedy_policy(q,self.state[state],action)
                a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
                next_state,reward,end=search_space[self.state[state]][action[a]]
                episode.append([state,a,reward])
                if end:
                    if self.save_episode==True:
                        _episode.append([self.state[state],action[a],reward,end])
                    break
                if self.save_episode==True:
                    _episode.append([self.state[state],action[a],reward])
                state=next_state
        else:
            for _ in range(self.episode_step):
                action_prob=self.epsilon_greedy_policy(q,self.state[state],action)
                a=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
                next_state,reward,end=search_space[self.state[state]][action[a]]
                episode.append([state,a,reward])
                if end:
                    if self.save_episode==True:
                        _episode.append([self.state[state],action[a],reward,end])
                    break
                if self.save_episode==True:
                    _episode.append([self.state[state],action[a],reward])
                state=next_state
        if self.save_episode==True:
            self.episode.append(_episode)
        return episode
    
    
    def first_visit(self,episode,q,r_sum,r_count,discount):
        state_action_set=set()
        delta=0
        self.delta=0
        for i,[state,action,reward] in enumerate(episode):
            state_action=(state,action)
            first_visit_index=i
            G=sum(np.power(discount,i)*x[2] for i,x in enumerate(episode[first_visit_index:]))
            if state_action not in state_action_set:
                state_action_set.add(state_action)
                if i==0:
                    r_sum[state_action]=G
                    r_count[state_action]=1
                else:
                    r_sum[state_action]+=G
                    r_count[state_action]+=1
                    delta+=np.abs(q[state][action]-r_sum[state_action]/r_count[state_action])
            q[state][action]=r_sum[state_action]/r_count[state_action]
        self.delta+=delta/len(episode)
        return q,r_sum,r_count
    
    
    def learn(self,episode_num,path=None,one=True):
        self.delta=0
        if len(self.state_list)>self.q.shape[0] or len(self.action)>self.q.shape[1]:
            q=self.q*tf.ones([len(self.state_list),len(self.action)],dtype=tf.float32)[:self.q.shape[0],:self.q.shape[1]]
            self.q=q.numpy()
        t1=time.time()
        for i in range(episode_num):
            s=np.random.choice(np.arange(len(self.state_list)),p=np.ones(len(self.state_list))*1/len(self.state_list))
            e=self.episode(self.q,self.state_list[s],self.action,self.search_space,self.episode_step)
            self.q,self.r_sum,self.r_count=self.first_visit(e,self.q,self.r_sum,self.r_count,self.discount)
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
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.r_sum)
        pickle.dump(self.r_count)
        pickle.dump(self.epsilon)
        pickle.dump(self.discount)
        pickle.dump(self.theta)
        pickle.dump(self.delta)
        pickle.dump(self.total_episode)
        pickle.dump(self.total_time)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.r_sum=pickle.load(input_file)
        self.r_count=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.theta=pickle.load(input_file)
        self.delta=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
