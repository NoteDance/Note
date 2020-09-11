import tensorflow as tf
from tensorflow.python.ops import state_ops
import numpy as np
import pickle
import time


class DQN:
    def __init__(self,predict_net,target_net,predict_p,target_p,state,action,search_space,epsilon=None,discount=None,memory_size=None,batch=None,update_step=None,optimizer=None,lr=None,save_episode=True):
        self.predict_net=predict_net
        self.target_net=target_net
        self.predict_p=predict_p
        self.target_p=target_p
        self.episode=[]
        self.state=state
        self.action=action
        self.search_space=search_space
        self.epsilon=epsilon
        self.discount=discount
        self.memory_size=memory_size
        self.batch=batch
        self.update_step=update_step
        self.lr=lr
        self.optimizer=optimizer(self.lr)
        self.save_episode=save_episode
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def epsilon_greedy_policy(self,state,action):
        action_prob=action[self.state[state]]
        action_prob=action_prob*self.epsilon/np.sum(action[self.state[state]])
        best_action=np.argmax(self.predict_net(self.state[self.state[state]]).numpy())
        action_prob[best_action]+=1-self.epsilon
        return action_prob
    
    
    def batch(self,j,index1,index2):
        random=np.arange(len(self.memory_state))
        np.random.shuffle(random)
        if j==0:
            self._memory_state=self.memory_state[random]
            self._memory_action=self.memory_action[random]
            self._memory_next_state=self.memory_next_state[random]
            self._memory_reward=self.memory_reward[random]
        if index1==self.batches*self.batch:
            return tf.concat([self._memory_state[index1:],self._memory_state[:index2]]),tf.concat([self._memory_action[index1:],self._memory_action[:index2]]),tf.concat([self._memory_next_state[index1:],self._memory_next_state[:index2]]),tf.concat([self._memory_reward[index1:],self._memory_reward[:index2]])
        else:
            return self._memory_state[index1:index2],self._memory_action[index1:index2],self._memory_next_state[index1:index2],self._memory_reward[index1:index2]
    
    
    def update_parameter(self):
        for i in range(len(self.predict_p)):
            state_ops.assign(self.target_p[i],self.predict_p[i])
        self.a=0
        return
    
    
    def loss(self,state,action,next_state,reward):
        return (reward+self.discount*tf.reduce_max(self.target_net(next_state),axis=-1)-self.predict_net(state)[np.arange(len(action)),action])**2
    
    
    def learn(self,episode_num,path=None,one=True):
        memory_state=[]
        memory_action=[]
        memory_next_state=[]
        memory_reward=[]
        for i in range(episode_num):
            self.a=0
            loss=0
            episode=[]
            state=np.random.choice(np.arange(len(self.state)),p=np.ones(len(self.state))*1/len(self.state))
            while True:
                t1=time.time()
                action_prob=self.epsilon_greedy_policy(state,self.action)
                action=np.random.choice(np.arange(action_prob.shape[0]),p=action_prob)
                next_state,reward,end=self.search_space[self.state[state]][self.action[action]]
                state=next_state
                if end:
                    if self.save_episode==True:
                        episode.append([self.state[state],self.action[action],reward,end])
                    break
                if self.save_episode==True:
                    episode.append([self.state[state],self.action[action],reward])
                self.a+=1
                memory_state.append(self.state[state])
                memory_action.append(action)
                memory_next_state.append(next_state)
                memory_reward.append(reward)
                if len(memory_state)>self.memory_size:
                    del memory_state[0]
                    del memory_action.pop[0]
                    del memory_next_state.pop[0]
                    del memory_reward.pop[0]
                self.memory_state=tf.Tensor(memory_state)
                self.memory_action=tf.Tensor(memory_action)
                self.memory_next_state=tf.Tensor(memory_next_state)
                self.memory_reward=tf.Tensor(memory_reward)
                if len(self.memory_state)<self.batch:
                    loss=self.loss(self.memory_state,self.memory_action,self.memory_next_state,self.memory_reward)
                else:
                    self.batches=int((len(self.memory_state)-len(self.memory_state)%self.batch)/self.batch)
                    for j in range(self.batches):
                        index1=j*self.batch
                        index2=(j+1)*self.batch
                        state_batch,action_batch,next_state_batch,reward_batch=self.batch(j,index1,index2)
                        batch_loss=self.loss(state_batch,action_batch,next_state_batch,reward_batch)
                        with tf.GradientTape() as tape:
                            gradient=tape.gradient(loss,self.predict_p)
                            self.optimizer.apply_gradients(zip(gradient,self.predict_p))
                        loss+=batch_loss
                    if len(self.memory_state)%self.batch!=0:
                        self.batches+=1
                        index1=self.batches*self.batch
                        index2=self.batch-(len(self.memory_state)-self.batches*self.batch)
                        state_batch,action_batch,next_state_batch,reward_batch=self.batch(j,index1,index2)
                        batch_loss=self.loss(state_batch,action_batch,next_state_batch,reward_batch)
                        with tf.GradientTape() as tape:
                            gradient=tape.gradient(loss,self.predict_p)
                            self.optimizer.apply_gradients(zip(gradient,self.predict_p))
                        loss+=batch_loss
                    loss=loss/self.batches
                t2=time.time()
                self.time+=(t2-t1)
                if self.a==self.update_step:
                    self.update_parameter()
                if episode_num%10!=0:
                    temp=episode_num-episode_num%10
                    temp=int(temp/10)
                else:
                    temp=episode_num/10
                if temp==0:
                    temp=1
                if i%temp==0:
                    print('episode_num:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if path!=None and i%episode_num*2==0:
                        self.save(path,i,one)
                self.episode_num+=1
                self.total_episode+=1
            if self.save_episode==True:
                self.episode.append(episode)
        if self.time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(loss))
        print('time:{0}s'.format(self.time))
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.memory_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.memory_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
