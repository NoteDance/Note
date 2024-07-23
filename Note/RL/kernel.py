import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import os
import time


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        if hasattr(self.nn,'km'):
            self.nn.km=1
        self.platform=None
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.episode_set=[]
        self.epsilon=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.trial_count=None
        self.criterion=None
        self.reward_list=[]
        self.suspend=False
        self.save_epi=None
        self.max_episode_count=None
        self.path=None
        self.avg_reward=None
        self.save_best_only=False
        self.save_param_only=False
        self.path_list=[]
        self.loss=None
        self.loss_list=[]
        self.sc=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def action_vec(self):
        if self.epsilon!=None:
            self.action_one=np.ones(self.genv.action_space.n,dtype=np.int8)
        return
    
    
    def init(self):
        try:
            if hasattr(self.nn,'pr'):
                self.nn.pr.TD=np.array(0)
        except Exception as e:
            raise e
        self.suspend=False
        self.save_epi=None
        self.episode_set=[]
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.reward_list=[]
        self.loss=0
        self.loss_list=[]
        self.sc=0
        self.total_episode=0
        self.time=0
        self.total_time=0
        return
    
    
    def set_up(self,epsilon=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None,HER=False):
        if epsilon!=None:
            self.epsilon=epsilon
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_step!=None:
            self.update_step=update_step
        if trial_count!=None:
            self.trial_count=trial_count
        if criterion!=None:
            self.criterion=criterion
        self.HER=HER
        self.action_vec()
        return
    
    
    def epsilon_greedy_policy(self,s):
        action_prob=self.action_one*self.epsilon/len(self.action_one)
        try:
            if hasattr(self.platform,'DType'):
                best_a=np.argmax(self.nn.nn.fp(s))
                action_prob[best_a]+=1-self.epsilon
            else:
                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                best_a=self.nn.nn(s).argmax()
                action_prob[best_a.numpy()]+=1-self.epsilon 
        except Exception as e:
            raise e
        return action_prob
    
    
    def run_agent(self, max_steps, seed=None):
        state_history = []

        steps = 0
        reward_ = 0
        if seed==None:
            state = self.nn.genv.reset()
        else:
            state = self.nn.genv.reset(seed=seed)
        for step in range(max_steps):
            if hasattr(self.platform,'DType'):
                if not hasattr(self, 'noise'):
                    action = np.argmax(self.nn.nn.fp(state))
                else:
                    action = self.nn.actor.fp(state).numpy()
            else:
                if not hasattr(self, 'noise'):
                    action = np.argmax(self.nn.nn.fp(state))
                else:
                    action = self.nn.actor.fp(state).detach().numpy()
            next_state, reward, done, _ = self.env.step(action)
            state_history.append(state)
            steps+=1
            reward_+=reward
            if done:
                break
            state = next_state
        
        return state_history,reward_,steps
    
    
    @tf.function(jit_compile=True)
    def tf_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        with self.platform.GradientTape(persistent=True) as tape:
            loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        if hasattr(self.nn,'gradient'):
            gradient=self.nn.gradient(tape,loss)
            if hasattr(self.nn.opt,'apply_gradients'):
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            else:
                self.nn.opt(gradient)
        else:
            if hasattr(self.nn,'nn'):
                gradient=tape.gradient(loss,self.nn.param)
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            else:
                actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                critic_gradient=tape.gradient(loss[1],self.nn.param[1])
                self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
        return loss
    
    
    def pytorch_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        self.nn.backward(loss)
        self.nn.opt()
        return loss
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        if hasattr(self.platform,'DType'):
            loss=self.tf_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        else:
            loss=self.pytorch_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        return loss
    
    
    def opt_ol(self,state,action,next_state,reward,done):
        if hasattr(self.platform,'DType'):
            loss=self.tf_opt(state,action,next_state,reward,done)
        else:
            loss=self.pytorch_opt(state,action,next_state,reward,done)
        return loss
    
    
    def pool(self,s,a,next_s,r,done):
        if type(self.state_pool)!=np.ndarray and self.state_pool==None:
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
    
    
    def choose_action(self,s):
        if hasattr(self.nn,'nn'):
            if hasattr(self.platform,'DType'):
                if self.epsilon==None:
                    self.epsilon=self.nn.epsilon(self.sc)
                if self.epsilon==None and hasattr(self.nn,'epsilon')!=True:
                    action_prob=self.nn.nn.fp(s)
                else:
                    action_prob=self.epsilon_greedy_policy(s)
                a=np.random.choice(self.action_count,p=action_prob)
            else:
                if self.epsilon==None:
                    self.epsilon=self.nn.epsilon(self.sc)
                if self.epsilon==None and hasattr(self.nn,'epsilon')!=True:
                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                    action_prob=self.nn.nn(s)
                else:
                    action_prob=self.epsilon_greedy_policy(s)
                a=np.random.choice(self.action_count,p=action_prob)
        else:
            if hasattr(self.nn,'action'):
                if hasattr(self.platform,'DType'):
                    if self.epsilon==None:
                        self.epsilon=self.nn.epsilon(self.sc)
                    a=self.nn.action(s).numpy()
                else:
                    if self.epsilon==None:
                        self.epsilon=self.nn.epsilon(self.sc)
                    a=self.nn.action(s).detach().numpy()
            else:
                if hasattr(self.platform,'DType'):
                    a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                else:
                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                    a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()
        return a
    
    
    def data_func(self):
        if hasattr(self.nn,'pr'):
            s,a,next_s,r,d=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch)
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
                reward, done = self.nn.reward_done_func(next_state, goal)
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
        
    
    def _train(self):
        if len(self.state_pool)<self.batch:
            return np.array(0.)
        else:
            loss=0
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if hasattr(self.nn,'pr') or self.HER==True:
                for j in range(batches):
                    self.suspend_func()
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                    loss+=batch_loss
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
                if len(self.state_pool)%self.batch!=0:
                    self.suspend_func()
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                    loss+=batch_loss
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
            else:
                train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                    self.suspend_func()
                    if hasattr(self.platform,'DType'):
                        pass
                    else:
                        state_batch=state_batch.numpy()
                        action_batch=action_batch.numpy()
                        next_state_batch=next_state_batch.numpy()
                        reward_batch=reward_batch.numpy()
                        done_batch=done_batch.numpy()
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                    loss+=batch_loss
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
            if self.update_step!=None:
                if self.sc%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
        if hasattr(self.platform,'DType'):
            loss=loss.numpy()/batches
        else:
            loss=loss.detach().numpy()/batches
        return loss
    
    
    def train_(self):
        episode=[]
        self.reward=0
        s=self.nn.env(initial=True)
        if hasattr(self.platform,'DType'):
            s=np.array(s)
        while True:
            s=np.expand_dims(s,axis=0)
            a=self.choose_action(s)
            next_s,r,done=self.nn.env(a)
            if hasattr(self.platform,'DType'):
                next_s=np.array(next_s)
                r=np.array(r)
                done=np.array(done)
            if hasattr(self.nn,'pool'):
                self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,r,done])
            else:
                self.pool(s,a,next_s,r,done)
            if hasattr(self.nn,'pr'):
                self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD)
                if len(self.state_pool)>self.pool_size:
                    TD=np.array(0)
                    self.nn.pr.TD=np.append(TD,self.nn.pr.TD[2:])
            self.reward=r+self.reward
            loss=self._train()
            self.sc+=1
            if done:
                self.reward_list.append(self.reward)
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
                return loss,episode,done
            s=next_s
        return loss,episode,done
    
    
    def train(self,episode_count,path=None,save_freq=1,max_save_files=None,p=None):
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if episode_count%10!=0:
            p=episode_count-episode_count%self.p
            p=int(p/self.p)
        else:
            p=episode_count/(self.p+1)
            p=int(p)
        if p==0:
            p=1
        self.max_save_files=max_save_files
        if episode_count!=None:
            for i in range(episode_count):
                t1=time.time()
                loss,episode,done=self.train_()
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if path!=None and i%save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(path)
                    else:
                        self.save_(path)
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
                            print('last loss:{0:.6f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                try:
                    try:
                        self.nn.ec.assign_add(1)
                    except Exception:
                        self.nn.ec+=1
                except Exception:
                    pass
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                loss,episode,done=self.train_()
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if path!=None and i%save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(path)
                    else:
                        self.save_(path)
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
                            print('last loss:{0:.6f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                if hasattr(self.nn,'ec'):
                    try:
                        self.nn.ec.assign_add(1)
                    except Exception:
                        self.nn.ec+=1
                t2=time.time()
                self.time+=(t2-t1)
        time_=self.time-int(self.time)
        if time_<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
        self.total_time+=self.time
        print('last loss:{0:.6f}'.format(loss))
        print('last reward:{0}'.format(self.reward))
        print()
        print('time:{0}s'.format(self.time))
        return
    
    
    def train_online(self):
        while True:
            if hasattr(self.nn,'save'):
                self.nn.save(self.save_)
            if hasattr(self.nn,'stop_flag'):
                if self.nn.stop_flag==True:
                    return
            if hasattr(self.nn,'stop_func'):
                if self.nn.stop_func():
                    return
            if hasattr(self.nn,'suspend_func'):
                self.nn.suspend_func()
            data=self.nn.online()
            if data=='stop':
                return
            elif data=='suspend':
                self.nn.suspend_func()
            loss=self.opt_ol(data[0],data[1],data[2],data[3],data[4])
            loss=loss.numpy()
            self.nn.train_loss_list.append(loss)
            if len(self.nn.train_acc_list)==self.nn.max_length:
                del self.nn.train_acc_list[0]
            if hasattr(self.nn,'counter'):
                self.nn.counter+=1
        return
        
    
    def suspend_func(self):
        if self.suspend==True:
            if self.save_epoch==None:
                print('Training have suspended.')
            else:
                self._save()
            while True:
                if self.suspend==False:
                    print('Training have continued.')
                    break
        return
    
    
    def _save(self):
        if self.save_epi==self.total_episode:
            self.save_()
            self.save_epi=None
            print('\nNeural network have saved and training have suspended.')
            return
        elif self.save_epi!=None and self.save_epi>self.total_episode:
            print('\nsave_epoch>total_epoch')
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
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
            if self.max_save_files==None:
                parameter_file=open(path,'wb')
            else:
                path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_episode))
                parameter_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self.nn.param,parameter_file)
            parameter_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save_param(path)
                        self.avg_reward=avg_reward
        return
    
    
    def save_param(self,path):
        parameter_file=open(path,'wb')
        pickle.dump(self.nn.param,parameter_file)
        parameter_file.close()
        return
    
    
    def restore_param(self,path):
        parameter_file=open(path,'rb')
        param=pickle.load(parameter_file)
        param_flat=nest.flatten(param)
        param_flat_=nest.flatten(self.nn.param)
        for i in range(len(param_flat)):
            state_ops.assign(param_flat_[i],param_flat[i])
        self.nn.param=nest.pack_sequence_as(self.nn.param,param_flat_)
        parameter_file.close()
        return
    
    
    def save_(self,path):
        if self.save_best_only==False:
            if self.max_save_files==None:
                output_file=open(path,'wb')
            else:
                path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_episode))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self.nn,output_file)
            pickle.dump(self.epsilon,output_file)
            pickle.dump(self.pool_size,output_file)
            pickle.dump(self.batch,output_file)
            pickle.dump(self.update_step,output_file)
            pickle.dump(self.reward_list,output_file)
            pickle.dump(self.loss,output_file)
            pickle.dump(self.loss_list,output_file)
            pickle.dump(self.sc,output_file)
            pickle.dump(self.total_episode,output_file)
            pickle.dump(self.total_time,output_file)
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
        pickle.dump(self.nn,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.loss,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.sc,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        if hasattr(self.nn,'km'):
            self.nn.km=1
        self.epsilon=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.sc=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
