from torch.utils.data import DataLoader
from Note.RL import rl
from Note.RL.rl.prioritized_replay import pr
from multiprocessing import Array,Value
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import statistics
import pickle
import os
import time


class RL:
    def __init__(self):
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.reward_list=[]
        self.step_counter=0
        self.prioritized_replay=pr()
        self.seed=7
        self.optimizer_=None
        self.path=None
        self.save_freq=1
        self.save_freq_=None
        self.max_save_files=None
        self.save_best_only=False
        self.save_param_only=False
        self.path_list=[]
        self.loss=None
        self.loss_list=[]
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def set_up(self,policy=None,noise=None,pool_size=None,batch=None,update_batches=None,update_steps=None,trial_count=None,criterion=None,PPO=False,HER=False,pr=False,epsilon=None,initial_TD=7,alpha=0.7):
        self.policy=policy
        self.noise=noise
        self.pool_size=pool_size
        self.batch=batch
        self.update_batches=update_batches
        self.update_steps=update_steps
        self.trial_count=trial_count
        self.criterion=criterion
        self.PPO=PPO
        self.HER=HER
        self.pr=pr
        self.epsilon=epsilon
        self.initial_TD=initial_TD
        self.alpha=alpha
        return
    
    
    def pool(self,s,a,next_s,r,done,index=None):
        if self.pool_network==True:
            if type(self.state_pool_list[index])!=np.ndarray and self.state_pool_list[index]==None:
                if type(s) in [int,float]:
                    s=np.array(s)
                    self.state_pool_list[index]=np.expand_dims(s,axis=0)
                elif type(s)==tuple:
                    s=np.array(s)
                    self.state_pool_list[index]=s
                else:
                    self.state_pool_list[index]=s
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool_list[index]=np.expand_dims(a,axis=0)
                else:
                    self.action_pool_list[index]=a
                self.next_state_pool_list[index]=np.expand_dims(next_s,axis=0)
                self.reward_pool_list[index]=np.expand_dims(r,axis=0)
                self.done_pool_list[index]=np.expand_dims(done,axis=0)
            else:
                self.state_pool_list[index]=np.concatenate((self.state_pool_list[index],s),0)
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool_list[index]=np.concatenate((self.action_pool_list[index],np.expand_dims(a,axis=0)),0)
                else:
                    self.action_pool_list[index]=np.concatenate((self.action_pool_list[index],a),0)
                self.next_state_pool_list[index]=np.concatenate((self.next_state_pool_list[index],np.expand_dims(next_s,axis=0)),0)
                self.reward_pool_list[index]=np.concatenate((self.reward_pool_list[index],np.expand_dims(r,axis=0)),0)
                self.done_pool_list[index]=np.concatenate((self.done_pool[7],np.expand_dims(done,axis=0)),0)
            if len(self.state_pool_list[index])>math.ceil(self.pool_size/self.processes):
                self.state_pool_list[index]=self.state_pool_list[index][1:]
                self.action_pool_list[index]=self.action_pool_list[index][1:]
                self.next_state_pool_list[index]=self.next_state_pool_list[index][1:]
                self.reward_pool_list[index]=self.reward_pool_list[index][1:]
                self.done_pool_list[index]=self.done_pool_list[index][1:]
        else:
            if type(self.state_pool)!=np.ndarray and self.state_pool==None:
                if type(s) in [int,float]:
                    s=np.array(s)
                    self.state_pool=np.expand_dims(s,axis=0)
                elif type(s)==tuple:
                    s=np.array(s)
                    self.state_pool=s
                else:
                    self.state_pool=s
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool=np.expand_dims(a,axis=0)
                else:
                    self.action_pool=a
                self.next_state_pool=np.expand_dims(next_s,axis=0)
                self.reward_pool=np.expand_dims(r,axis=0)
                self.done_pool=np.expand_dims(done,axis=0)
            else:
                self.state_pool=np.concatenate((self.state_pool,s),0)
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0)
                else:
                    self.action_pool=np.concatenate((self.action_pool,a),0)
                self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(next_s,axis=0)),0)
                self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0)
                self.done_pool=np.concatenate((self.done_pool,np.expand_dims(done,axis=0)),0)
            if len(self.state_pool)>self.pool_size:
                self.state_pool=self.state_pool[1:]
                self.action_pool=self.action_pool[1:]
                self.next_state_pool=self.next_state_pool[1:]
                self.reward_pool=self.reward_pool[1:]
                self.done_pool=self.done_pool[1:]
        return
    
    
    def select_action(self,s):
        if self.policy!=None:
            output=self.action(s).detach().numpy()
            if isinstance(self.policy, rl.SoftmaxPolicy):
                a=self.policy.select_action(len(output), output)
            elif isinstance(self.policy, rl.EpsGreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.GreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.MaxBoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannGumbelQPolicy):
                if self.pool_network==True:
                    a=self.policy.select_action(output, self.step_counter.value)
                else:
                    a=self.policy.select_action(output, self.step_counter)
        elif self.noise!=None:
            a=(self.action(s)+self.noise.sample()).detach().numpy()
        return a
    
    
    def env_(self,a=None,initial=None,p=None):
        if initial==True:
            if self.pool_network==True:
                state=self.env[p].reset(seed=self.seed)
                return state
            else:
                state=self.env.reset(seed=self.seed)
                return state 
        else:
            if self.pool_network==True:
                next_state,reward,done,_=self.env[p].step(a)
                return next_state,reward,done
            else:
                next_state,reward,done,_=self.env.step(a)
                return next_state,reward,done
    
    
    def data_func(self):
        if self.pr_:
            s,a,next_s,r,d=self.prioritized_replay.sample(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.epsilon,self.alpha,self.batch)
        elif self.HER:
            s = []
            a = []
            next_s = []
            r = []
            d = []
            for _ in range(self.batch):
                step_state = np.random.randint(0, len(self.state_pool)-1)
                step_goal = np.random.randint(step_state+1, step_state+np.argmax(self.done_pool[step_state+1:])+2)
                state = self.state_pool[step_state]
                next_state = self.next_state_pool[step_state]
                action = self.action_pool[step_state]
                goal = self.state_pool[step_goal]
                reward, done = self.reward_done_func(next_state, goal)
                state = np.hstack((state, goal))
                next_state = np.hstack((next_state, goal))
                s.append(state)
                a.append(action)
                next_s.append(next_state)
                r.append(reward)
                d.append(done)
            s = np.array(s)
            a = np.array(a)
            next_s = np.array(next_s)
            r = np.array(r)
            d = np.array(d)
        return s,a,next_s,r,d
    
    
    def train_step(self, train_data, optimizer):
        loss = self.__call__(*train_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    
    
    def train1(self, optimizer):
        if len(self.state_pool)<self.batch:
            if self.loss!=None:
                return self.loss
            else:
                return np.array(0.)
        else:
            loss=0
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if self.pr==True  or self.HER==True:
                for j in range(batches):
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer)
                    self.batch_counter+=1
                if len(self.state_pool)%self.batch!=0:
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer)
                    self.batch_counter+=1
            else:
                if self.pool_network==True:
                    if self.shuffle!=True:
                        train_ds=DataLoader((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool),batch_size=self.batch)
                    else:
                        train_ds=DataLoader((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool),batch_size=self.batch,shuffle=True)
                else:
                    train_ds=DataLoader((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool),batch_size=self.batch,shuffle=True)
                for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                    loss+=self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer)
                    self.batch_counter+=1
            if self.update_steps!=None:
                if self.pool_network==True:
                    if self.batch_counter%self.update_batches==0:
                        self.update_param()
                        if self.PPO:
                            self.state_pool=None
                            self.action_pool=None
                            self.next_state_pool=None
                            self.reward_pool=None
                            self.done_pool=None
                else:
                    if self.step_counter%self.update_steps==0:
                        self.update_param()
                        if self.PPO:
                            self.state_pool=None
                            self.action_pool=None
                            self.next_state_pool=None
                            self.reward_pool=None
                            self.done_pool=None
            else:
                self.update_param()
                if self.PPO:
                    self.state_pool=None
                    self.action_pool=None
                    self.next_state_pool=None
                    self.reward_pool=None
                    self.done_pool=None
        return loss.detach().numpy()/batches
    
    
    def train2(self, optimizer):
        self.reward=0
        s=self.env_(initial=True)
        s=np.array(s)
        while True:
            s=np.expand_dims(s,axis=0)
            a=self.select_action(s)
            next_s,r,done=self.env_(a)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            self.pool(s,a,next_s,r,done)
            if self.pr==True:
                self.prioritized_replay.TD=np.append(self.prioritized_replay.TD,self.initial_TD)
                if len(self.state_pool)>self.pool_size:
                    TD=np.array(0)
                    self.prioritized_replay.TD=np.append(TD,self.prioritized_replay.TD[2:])
            self.reward=r+self.reward
            loss=self.train1(optimizer)
            self.step_counter+=1
            if done:
                self.reward_list.append(self.reward)
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
                return loss
            s=next_s
            
            
    def store_in_parallel(self,p):
        self.reward[p]=0
        s=self.env_(initial=True,p=p)
        s=np.array(s)
        while True:
            if None not in self.state_pool_list:
                inverse_len=[1/len(state_pool) for state_pool in self.state_pool_list]
                inverse_len=np.array(inverse_len)
                total_inverse=np.sum(inverse_len)
                prob=inverse_len/total_inverse
                index=np.random.choice(self.processes,p=prob)
            else:
                index=np.random.choice(self.processes)
            s=np.expand_dims(s,axis=0)
            a=self.select_action(s)
            next_s,r,done=self.env_(a,p=p)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            self.lock_list[index].acquire()
            self.pool(s,a,next_s,r,done,index)
            self.step_counter.value+=1
            self.lock_list[index].release()
            self.reward[p]=r+self.reward[p]
            if done:
                self.reward_list.append(self.reward[p])
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
            s=next_s
    
    
    def fit(self, optimizer, episodes=None, mp=None, manager=None, processes=None, shuffle=False, p=None):
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if episodes%10!=0:
            p=episodes-episodes%self.p
            p=int(p/self.p)
        else:
            p=episodes/(self.p+1)
            p=int(p)
        if p==0:
            p=1
        self.pool_network=False
        self.processes=processes
        self.shuffle=shuffle
        if mp!=None:
            self.pool_network=True
            self.state_pool_list=manager.list()
            self.action_pool_list=manager.list()
            self.next_state_pool_list=manager.list()
            self.reward_pool_list=manager.list()
            self.done_pool_list=manager.list()
            for _ in range(processes):
                self.state_pool_list.append(None)
                self.action_pool_list.append(None)
                self.next_state_pool_list.append(None)
                self.reward_pool_list.append(None)
                self.done_pool_list.append(None)
            self.reward=np.zeros(processes,dtype='float32')
            self.reward=Array('f',self.reward)
            self.reward_list=manager.list([])
            self.step_counter=Value('i',0)
            self.lock_list=[mp.Lock() for _ in range(processes)]
        self.optimizer_=optimizer
        if episodes!=None:
            for i in range(episodes):
                t1=time.time()
                if self.pool_network==True:
                    process_list=[]
                    for p in range(processes):
                        process=mp.Process(target=self.store_in_parallel,args=(p,))
                        process.start()
                        process_list.append(process)
                    for process in process_list:
                        process.join()
                    self.state_pool=np.array(self.state_pool_list)
                    self.action_pool=np.array(self.action_pool_list)
                    self.next_state_pool=np.array(self.next_state_pool_list)
                    self.reward_pool=np.array(self.reward_pool_list)
                    self.done_pool=np.array(self.done_pool_list)
                    loss=self.train1(self.optimizer_)
                else:
                    loss=self.train2(self.optimizer_)
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.4f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                if self.pool_network==True:
                    process_list=[]
                    for p in range(processes):
                        process=mp.Process(target=self.store_in_parallel,args=(p,))
                        process.start()
                        process_list.append(process)
                    for process in process_list:
                        process.join()
                    self.state_pool=np.array(self.state_pool_list)
                    self.action_pool=np.array(self.action_pool_list)
                    self.next_state_pool=np.array(self.next_state_pool_list)
                    self.reward_pool=np.array(self.reward_pool_list)
                    self.done_pool=np.array(self.done_pool_list)
                    loss=self.train1(self.optimizer_)
                else:
                    loss=self.train2(self.optimizer_)
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.4f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        time_=self.time-int(self.time)
        if time_<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
        self.total_time+=self.time
        print('last loss:{0:.4f}'.format(loss))
        print('last reward:{0}'.format(self.reward))
        print()
        print('time:{0}s'.format(self.time))
        return
    
    
    def run_agent(self, max_steps, seed=None):
        state_history = []

        steps = 0
        reward_ = 0
        if seed==None:
            state = self.env.reset()
        else:
            state = self.env.reset(seed=seed)
        for step in range(max_steps):
            if self.noise==None:
                action = np.argmax(self.action(state))
            else:
                action = self.action(state).numpy()
            next_state, reward, done, _ = self.env.step(action)
            state_history.append(state)
            steps+=1
            reward_+=reward
            if done:
                break
            state = next_state
        
        return state_history,reward_,steps
    
    
    def animate_agent(self, max_steps, mode='rgb_array', save_path=None, fps=None, writer='imagemagick'):
        state_history,reward,steps = self.run_agent(max_steps)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        self.env.reset()
        img = ax.imshow(self.env.render(mode=mode))

        def update(frame):
            img.set_array(self.env.render(mode=mode))
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=state_history, blit=True)
        plt.show()
        
        print('steps:{0}'.format(steps))
        print('reward:{0}'.format(reward))
        
        if save_path!=None:
            ani.save(save_path, writer=writer, fps=fps)
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('reward:{0:.4f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('loss:{0:.4f}'.format(self.loss_list[-1]))
        return
    
    
    def visualize_reward_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list,'r-',label='reward')
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list,'b-',label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('reward and loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        return
    
    
    def save_param_(self,path):
        if self.save_best_only==False:
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(path,'wb')
            else:
                if self.train_acc!=None and self.test_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch,self.train_acc,self.test_acc))
                elif self.train_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.train_acc))
                else:
                    path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_epoch))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self.param,output_file)
            output_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save_param(path)
                        self.avg_reward=avg_reward
        return
    
    
    def save_param(self,path):
        output_file=open(path,'wb')
        pickle.dump(self.param,output_file)
        output_file.close()
        return
    
    
    def restore_param(self,path):
        input_file=open(path,'rb')
        param=pickle.load(input_file)
        for target_param,source_param in zip(self.param,param):
            target_param.data=source_param.data
        input_file.close()
        return
    
    
    def save_(self,path):
        if self.save_best_only==False:
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(path,'wb')
            else:
                if self.train_acc!=None and self.test_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch,self.train_acc,self.test_acc))
                elif self.train_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.train_acc))
                else:
                    path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_epoch))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self,output_file)
            output_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save(path)
                        self.avg_reward=avg_reward
        return
    
    
    def save(self,path):
        output_file=open(path,'wb')
        pickle.dump(self,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        model=pickle.load(input_file)
        self.__dict__.update(model.__dict__)
        input_file.close()
        return
