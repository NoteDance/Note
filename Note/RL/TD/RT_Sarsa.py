import tensorflow as tf
import numpy as np


class RT_Sarsa:
    def __init__(self,q,state,state_list,action,search_space,epsilon,alpha,discount):
        self.q=q
        self.state=state
        self.state_list=state_list
        self.action=action
        self.search_space=search_space
        self.epsilon=epsilon
        self.alpha=alpha
        self.discount=discount


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
    
    
    def RT_update_q(self,q,state):
        while True:
            action_prob=self.epsilon_greedy_policy(q,self.state[state],self.action)
            action=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
            next_state,reward,end=self.search_space[self.action[action]]
            q,next_state=self.td(q,reward,state,next_state,action)
            state=next_state
        return
    
    
    def learn(self):
        if len(self.state_list)>self.q.shape[0] or len(self.action)>self.q.shape[1]:
            q=self.q*tf.ones([len(self.state_list),len(self.action)],dtype=tf.float32)[:self.q.shape[0],:self.q.shape[1]]
            self.q=q.numpy()
        s=np.random.choice(np.arange(len(self.state_list)),p=np.ones(len(self.state_list))*1/len(self.state_list))
        self.q=self.update_q(self.q,self.state_list[s])
        return
