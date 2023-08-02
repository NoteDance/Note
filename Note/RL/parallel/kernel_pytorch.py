import torch
from multiprocessing import Value,Array
from Note.nn.parallel.assign_device_pytorch import assign_device
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import os


class kernel:
    def __init__(self,nn=None,process=None,device='GPU'):
        self.nn=nn
        if process!=None:
            self.reward=np.zeros(process,dtype=np.float32)
            self.sc=np.zeros(process,dtype=np.int32)
        self.device=device
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.episode=None
        self.batch=None
        self.update_step=None
        self.trial_count=None
        self.process=process
        self.priority_flag=False
        self.max_opt=None
        self.stop=False
        self.s=None
        self.filename='save.dat'
    
    
    def init(self,manager):
        self.state_pool=manager.dict({})
        self.action_pool=manager.dict({})
        self.next_state_pool=manager.dict({})
        self.reward_pool=manager.dict({})
        self.done_pool=manager.dict({})
        self.reward=Array('f',self.reward)
        self.loss=np.zeros(self.process)
        self.loss=Array('f',self.loss)
        self.sc=Array('i',self.sc)
        self.process_counter=Value('i',0)
        self.probability_list=manager.list([])
        self.running_flag_list=manager.list([])
        self.finish_list=manager.list([])
        self.running_flag=manager.list([0])
        self.reward_list=manager.list([])
        self.loss_list=manager.list([])
        self.episode_counter=Value('i',0)
        self.total_episode=Value('i',0)
        self.priority_p=Value('i',0)
        if self.priority_flag==True:
            self.opt_counter=Array('i',np.zeros(self.process,dtype=np.int32))
        try:
            self.nn.opt_counter=manager.list([self.nn.opt_counter])  
        except Exception:
            self.opt_counter_=manager.list()
        try:
            self.nn.ec=manager.list([self.nn.ec])  
        except Exception:
            self.ec_=manager.list()
        try:
            self.nn.bc=manager.list([self.nn.bc])
        except Exception:
            self.bc_=manager.list()
        self.episode_=Value('i',self.total_episode.value)
        self.stop_flag=Value('b',False)
        self.save_flag=Value('b',False)
        self.file_list=manager.list([])
        return
    
    
    def init_online(self,manager):
        self.nn.train_loss_list=manager.list([])
        self.nn.counter=manager.list([])
        self.nn.exception_list=manager.list([])
        return
    
    
    def action_vec(self):
        self.action_one=np.ones(self.action_count,dtype=np.int8)
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
        if epsilon!=None:
            self.epsilon=np.ones(self.process)*epsilon
        if episode_step!=None:
            self.episode_step=episode_step
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
        if epsilon!=None:
            self.action_vec()
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon,p):
        action_prob=self.action_one*epsilon/len(self.action_one)
        s=torch.tensor(s,dtype=torch.float).to(assign_device(p,self.device))
        best_a=self.nn.nn(s).argmax()
        action_prob[best_a.numpy()]+=1-epsilon 
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        pool_lock[index].acquire()
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None:
                self.state_pool[index]=s
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool[index]=np.expand_dims(a,axis=0)
                else:
                    self.action_pool[index]=a
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0)
                self.reward_pool[index]=np.expand_dims(r,axis=0)
                self.done_pool[index]=np.expand_dims(done,axis=0)
            else:
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)
                    if type(a)==int:
                        a=np.array(a)
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                    else:
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                except Exception:
                    pass
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size:
                self.state_pool[index]=self.state_pool[index][1:]
                self.action_pool[index]=self.action_pool[index][1:]
                self.next_state_pool[index]=self.next_state_pool[index][1:]
                self.reward_pool[index]=self.reward_pool[index][1:]
                self.done_pool[index]=self.done_pool[index][1:]
        except Exception:
            pool_lock[index].release()
            return
        pool_lock[index].release()
        return
    
    
    def get_index(self,p,lock):
        while len(self.running_flag_list)<p:
            pass
        if len(self.running_flag_list)==p:
            lock[0].acquire()
            self.running_flag_list.append(self.running_flag[1:].copy())
            lock[0].release()
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value:
            self.running_flag_list[p]=self.running_flag[1:].copy()
        while len(self.probability_list)<p:
            pass
        if len(self.probability_list)==p:
            lock[0].acquire()
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]))
            lock[0].release()
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])
        while True:
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p])
            if index in self.finish_list:
                continue
            else:
                break
        return index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):
        if hasattr(self.nn,'nn'):
            s=np.expand_dims(s,axis=0)
            action_prob=self.epsilon_greedy_policy(s,epsilon,p)
            a=np.random.choice(self.action_count,p=action_prob)
        else:
            if hasattr(self.nn,'action'):
                s=np.expand_dims(s,axis=0)
                a=self.nn.action(s,p).numpy()
            else:
                s=np.expand_dims(s,axis=0)
                s=torch.tensor(s,dtype=torch.float).to(assign_device(p,self.device))
                a=(self.nn.actor(s)+self.nn.noise()).numpy()
        next_s,r,done=self.nn.env(a,p)
        index=self.get_index(p,lock)
        next_s=np.array(next_s)
        r=np.array(r)
        done=np.array(done)
        self.pool(s,a,next_s,r,done,pool_lock,index)
        return next_s,r,done,index
    
    
    def end(self):
        if self.trial_count!=None:
            if len(self.reward_list)>=self.trial_count:
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                if self.criterion!=None and avg_reward>=self.criterion:
                    return True
        return False
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)
        if self.priority_flag==True and self.priority_p.value!=-1:
            while True:
                if self.stop_flag.value==True:
                    return None
                if p==self.priority_p.value:
                    break
                else:
                    continue
        if self.stop_func_():
            return None
        loss=loss.clone()
        self.nn.backward(loss,p)
        self.nn.opt(p)
        return loss
    
    
    def _train(self,p,j,batches,length):
        if j==batches-1:
            index1=batches*self.batch
            index2=self.batch-(length-batches*self.batch)
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0)
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0)
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0)
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0)
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0)
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)
            self.loss[p]+=loss
            if hasattr(self.nn,'bc'):
                bc=self.nn.bc[0]
                bc.assign_add(1)
                self.nn.bc[0]=bc
        else:
            index1=j*self.batch
            index2=(j+1)*self.batch
            state_batch=self.state_pool[p][index1:index2]
            action_batch=self.action_pool[p][index1:index2]
            next_state_batch=self.next_state_pool[p][index1:index2]
            reward_batch=self.reward_pool[p][index1:index2]
            done_batch=self.done_pool[p][index1:index2]
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)
            self.loss[p]+=loss
            if hasattr(self.nn,'bc'):
                bc=self.nn.bc[0]
                bc.assign_add(1)
                self.nn.bc[0]=bc
        return
    
    
    def train_(self,p):
        if len(self.done_pool[p])<self.batch:
            return
        else:
            self.loss[p]=0
            length=len(self.done_pool[p])
            batches=int((length-length%self.batch)/self.batch)
            if length%self.batch!=0:
                batches+=1
            for j in range(batches):
                if self.priority_flag==True:
                    self.priority_p.value=np.argmax(self.opt_counter)
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt:
                        self.priority_p.value=int(self.priority_p.value)
                    elif self.max_opt==None:
                        self.priority_p.value=int(self.priority_p.value)
                    else:
                        self.priority_p.value=-1
                if self.priority_flag==True:
                    self.opt_counter[p]=0
                if hasattr(self.nn,'attenuate'):
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter[p]=0
                    self.nn.opt_counter[0]=opt_counter
                self._train(p,j,batches,length)
                if self.priority_flag==True:
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')
                    opt_counter+=1
                if hasattr(self.nn,'attenuate'):
                    opt_counter=self.nn.opt_counter[0]
                    opt_counter+=1
                    self.nn.opt_counter[0]=opt_counter
            if self.update_step!=None:
                if self.sc[p]%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
            self.loss[p]=self.loss[p]/batches
        self.sc[p]+=1
        if hasattr(self.nn,'ec'):
            ec=self.nn.ec[0]
            ec.assign_add(1)
            self.nn.ec[0]=ec
        return
    
    
    def train(self,p,lock,pool_lock):
        lock[1].acquire()
        self.state_pool[p]=None
        self.action_pool[p]=None
        self.next_state_pool[p]=None
        self.reward_pool[p]=None
        self.done_pool[p]=None
        self.running_flag.append(1)
        self.process_counter.value+=1
        self.finish_list.append(None)
        lock[1].release()
        try:
            epsilon=self.epsilon[p]
        except Exception:
            epsilon=None
        while True:
            if self.episode_counter.value>=self.episode:
                break
            s=self.nn.env(p=p,initial=True)
            s=np.array(s)
            if self.episode_step==None:
                while True:
                    if self.episode_counter.value>=self.episode:
                        break
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)
                    self.reward[p]+=r
                    s=next_s
                    if type(self.done_pool[p])==np.ndarray:
                        self.train_(p)
                    if done:
                        self.episode_counter.value+=1
                        if len(lock)==4:
                            lock[3].acquire()
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if len(lock)==4:
                            lock[3].release()
                        break
            else:
                for l in range(self.episode_step):
                    if self.episode_counter.value>=self.episode:
                        break
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)
                    self.reward[p]+=r
                    s=next_s
                    if type(self.done_pool[p])==np.ndarray:
                        self.train_(p)
                    if done:
                        self.episode_counter.value+=1
                        if len(lock)==4:
                            lock[3].acquire()
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if len(lock)==4:
                            lock[3].release()
                        break
                    if l==self.episode_step-1:
                        self.episode_counter.value+=1
                        if len(lock)==4:
                            lock[3].acquire()
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if len(lock)==4:
                            lock[3].release()
            if len(lock)==3 or len(lock)==4:
                lock[2].acquire()
            self.save_()
            self.reward_list.append(self.reward[p])
            self.reward[p]=0
            if len(lock)==3 or len(lock)==4:
                lock[2].release()
        self.running_flag[p+1]=0
        if p not in self.finish_list:
            self.finish_list[p]=p
        lock[1].acquire()
        self.process_counter.value-=1
        lock[1].release()
        del self.state_pool[p]
        del self.action_pool[p]
        del self.next_state_pool[p]
        del self.reward_pool[p]
        del self.done_pool[p]
        return
    
    
    def train_online(self,p,lock=None,g_lock=None):
        if hasattr(self.nn,'counter'):
            self.nn.counter.append(0)
        while True:
            if hasattr(self.nn,'save'):
                self.nn.save(self.save,p)
            if hasattr(self.nn,'stop_flag'):
                if self.nn.stop_flag==True:
                    return
            if hasattr(self.nn,'stop_func'):
                if self.nn.stop_func(p):
                    return
            if hasattr(self.nn,'suspend_func'):
                self.nn.suspend_func(p)
            try:
                data=self.nn.online(p)
            except Exception as e:
                self.nn.exception_list[p]=e
            if data=='stop':
                return
            elif data=='suspend':
                self.nn.suspend_func(p)
            try:
                loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock,g_lock)
            except Exception as e:
                self.nn.exception_list[p]=e
            loss=loss.numpy()
            if len(self.nn.train_loss_list)==self.nn.max_length:
                del self.nn.train_loss_list[0]
            self.nn.train_loss_list.append(loss)
            try:
                if hasattr(self.nn,'counter'):
                    count=self.nn.counter[p]
                    count+=1
                    self.nn.counter[p]=count
            except IndexError:
                self.nn.counter.append(0)
                count=self.nn.counter[p]
                count+=1
                self.nn.counter[p]=count
        return
    
    
    def stop_func(self):
        if self.end():
            self.save(self.total_episode)
            self.save_flag.value=True
            self.stop_flag.value=True
            return True
        return False
    
    
    def stop_func_(self):
        if self.stop==True:
            if self.stop_flag.value==True or self.stop_func():
                return True
        return False
    
    
    def save_(self):
        if self.s!=None:
            if self.s==1:
                s_=1
            else:
                s_=self.s-1
            if self.total_episode.value%10!=0:
                s=self.total_episode.value-self.total_episode.value%s_
                s=int(s/s_)
                if s==0:
                    s=1
            else:
                s=self.total_episode.value/(s_+1)
                s=int(s)
                if s==0:
                    s=1
            if self.episode_.value%s==0:
                if self.saving_one==True:
                    self.save(self.total_episode.value)
                else:
                    self.save(self.total_episode.value,False)
        self.episode_.value+=1
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_list)),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.loss_list)),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def visualize_reward_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_list)),self.reward_list,'r-',label='reward')
        plt.plot(np.arange(len(self.loss_list)),self.loss_list,'b-',label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('reward and loss')
        return
    
    
    def save(self,i=None,one=True):
        if self.save_flag.value==True:
            return
        if one==True:
            output_file=open(self.filename,'wb')
        else:
            filename=self.filename.replace(self.filename[self.filename.find('.'):],'-{0}.dat'.format(i))
            output_file=open(filename,'wb')
            self.file_list.append([filename])
            if len(self.file_list)>self.s:
                os.remove(self.file_list[0][0])
                del self.file_list[0]
        self.update_nn_param()
        if hasattr(self.nn,'opt_counter'):
            self.nn.opt_counter=self.nn.opt_counter[0] 
        if hasattr(self.nn,'ec'):
            self.nn.ec=self.nn.ec[0]
        if hasattr(self.nn,'bc'):
            self.nn.bc=self.nn.bc[0]
        pickle.dump(self.nn,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(np.array(self.sc,dtype=np.int32),output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(list(self.reward_list),output_file)
        pickle.dump(list(self.loss_list),output_file)
        pickle.dump(self.total_episode.value,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        if hasattr(self.nn,'km'):
            self.nn.km=1
        if hasattr(self.nn,'opt_counter'):
            self.nn.opt_counter=self.opt_counter_
            self.nn.opt_counter.append(self.nn.opt_counter)
        if hasattr(self.nn,'ec'):
            self.nn.ec=self.ec_
            self.nn.ec.append(self.nn.ec)
        if hasattr(self.nn,'bc'):
            self.nn.bc=self.bc_
            self.nn.bc.append(self.nn.bc)
        self.epsilon=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.sc=pickle.load(input_file)
        self.sc=Array('i',self.sc)
        self.update_step=pickle.load(input_file)
        self.reward_list[:]=pickle.load(input_file)
        self.loss_list[:]=pickle.load(input_file)
        self.total_episode.value=pickle.load(input_file)
        self.episode_.value=self.total_episode.value
        input_file.close()
        return
