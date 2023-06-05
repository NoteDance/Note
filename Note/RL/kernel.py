import tensorflow as tf
from multiprocessing import Value,Array
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import os


class kernel:
    def __init__(self,nn=None,process=None):
        self.nn=nn
        if process!=None:
            self.reward=np.zeros(process,dtype=np.float32)
            self.loss=np.zeros(process,dtype=np.float32)
            self.sc=np.zeros(process,dtype=np.float32)
        self.state_pool={}
        self.action_pool={}
        self.next_state_pool={}
        self.reward_pool={}
        self.done_pool={}
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.episode=0
        self.update_step=None
        self.trial_count=None
        self.process=process
        self.process_counter=0
        self.probability_list=[]
        self.running_flag_list=[]
        self.finish_list=[]
        self.running_flag=[]
        self.PO=None
        self.priority_flag=False
        self.priority_p=0
        self.max_opt=None
        self.stop=False
        self.save_flag=False
        self.stop_flag=False
        self.filename='save.dat'
        self.reward_list=[]
        self.loss_list=[]
        self.total_episode=0
    
    
    def init(self,manager):
        self.state_pool=manager.dict(self.state_pool)
        self.action_pool=manager.dict(self.action_pool)
        self.next_state_pool=manager.dict(self.next_state_pool)
        self.reward_pool=manager.dict(self.reward_pool)
        self.done_pool=manager.dict(self.done_pool)
        self.reward=Array('f',self.reward)
        self.loss=Array('f',self.loss)
        self.sc=Array('f',self.sc)
        self.process_counter=Value('i',self.process_counter)
        self.probability_list=manager.list(self.probability_list)
        self.running_flag_list=manager.list(self.running_flag_list)
        self.finish_list=manager.list(self.finish_list)
        self.running_flag=manager.list([0])
        self.reward_list=manager.list(self.reward_list)
        self.loss_list=manager.list(self.loss_list)
        self.total_episode=Value('i',self.total_episode)
        self.priority_p=Value('i',self.priority_p)
        if self.priority_flag==True:
            self.opt_counter=Array('i',np.zeros(self.process,dtype=np.int32))
        try:
            if self.nn.attenuate!=None:
              self.nn.opt_counter=manager.list([self.nn.opt_counter])  
        except AttributeError:   
            pass
        try:
            self.nn.ec=manager.list(self.nn.ec)
        except AttributeError:
            pass
        try:
            self.nn.bc=manager.list(self.nn.bc)
        except AttributeError:
            pass
        self.stop_flag=Value('b',0)
        self.save_flag=Value('b',0)
        self.file_list=manager.list([])
        self.param=manager.dict()
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
    
    
    def epsilon_greedy_policy(self,s,epsilon):
        action_prob=self.action_one*epsilon/len(self.action_one)
        best_a=np.argmax(self.nn.nn.fp(s))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        pool_lock[index].acquire()
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None:
                self.state_pool[index]=s
                if type(a)==int:
                    if type(self.nn.param[0])!=list:
                        a=np.array(a,self.nn.param[0].dtype.name)
                    else:
                        a=np.array(a,self.nn.param[0][0].dtype.name)
                    self.action_pool[index]=np.expand_dims(a,axis=0)
                else:
                    if type(self.nn.param[0])!=list:
                        a=a.astype(self.nn.param[0].dtype.name)
                    else:
                        a=a.astype(self.nn.param[0][0].dtype.name)
                    self.action_pool[index]=a
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0)
                self.reward_pool[index]=np.expand_dims(r,axis=0)
                self.done_pool[index]=np.expand_dims(done,axis=0)
            else:
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)
                    if type(a)==int:
                        if type(self.nn.param[0])!=list:
                            a=np.array(a,self.nn.param[0].dtype.name)
                        else:
                            a=np.array(a,self.nn.param[0][0].dtype.name)
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                    else:
                        if type(self.nn.param[0])!=list:
                            a=a.astype(self.nn.param[0].dtype.name)
                        else:
                            a=a.astype(self.nn.param[0][0].dtype.name)
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                except:
                    pass
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size:
                self.state_pool[index]=self.state_pool[index][1:]
                self.action_pool[index]=self.action_pool[index][1:]
                self.next_state_pool[index]=self.next_state_pool[index][1:]
                self.reward_pool[index]=self.reward_pool[index][1:]
                self.done_pool[index]=self.done_pool[index][1:]
        except:
            pool_lock[index].release()
            return
        pool_lock[index].release()
        return
    
    
    def get_index(self,p,lock):
        while len(self.running_flag_list)<p:
            pass
        if len(self.running_flag_list)==p:
            if self.PO==1:
                lock[2].acquire()
            else:
                lock[3].acquire()
            self.running_flag_list.append(self.running_flag[1:].copy())
            if self.PO==1:
                lock[2].release()
            else:
                lock[3].release()
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value:
            self.running_flag_list[p]=self.running_flag[1:].copy()
        while len(self.probability_list)<p:
            pass
        if len(self.probability_list)==p:
            if self.PO==1:
                lock[2].acquire()
            else:
                lock[3].acquire()
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]))
            if self.PO==1:
                lock[2].release()
            else:
                lock[3].release()
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])
        while True:
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p])
            if index in self.finish_list:
                continue
            else:
                break
        return index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):
        try:
            if self.nn.nn!=None:
                s=np.expand_dims(s,axis=0)
                action_prob=self.epsilon_greedy_policy(s,epsilon)
                a=np.random.choice(self.action_count,p=action_prob)
                next_s,r,done=self.nn.env(a,p)
        except AttributeError:
            try:
                if self.nn.action!=None:
                    s=np.expand_dims(s,axis=0)
                    a=self.nn.action(s).numpy()
            except AttributeError:
                s=np.expand_dims(s,axis=0)
                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                next_s,r,done=self.nn.env(a,p)
        index=self.get_index(p,lock)
        if type(self.nn.param[0])!=list:
            next_s=np.array(next_s,self.nn.param[0].dtype.name)
            r=np.array(r,self.nn.param[0].dtype.name)
            done=np.array(done,self.nn.param[0].dtype.name)
        else:
            next_s=np.array(next_s,self.nn.param[0][0].dtype.name)
            r=np.array(r,self.nn.param[0][0].dtype.name)
            done=np.array(done,self.nn.param[0][0].dtype.name)
        self.pool(s,a,next_s,r,done,pool_lock,index)
        return next_s,r,done,index
    
    
    def end(self):
        if self.trial_count!=None:
            if len(self.reward_list)>=self.trial_count:
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                if self.criterion!=None and avg_reward>=self.criterion:
                    return True
    
    
    @tf.function(jit_compile=True)
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock=None,ln=None):
        with tf.GradientTape(persistent=True) as tape:
            try:
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            except TypeError:
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)
        if self.PO==1:
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return 0
            try:
                gradient=self.nn.gradient(tape,loss)
                try:
                    if self.nn.attenuate!=None:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)
                except AttributeError:
                    pass
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                        try:
                            if self.nn.attenuate!=None:
                                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)
                        except AttributeError:
                            pass
                except AttributeError:
                        actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                        critic_gradient=tape.gradient(loss[1],self.nn.param[1])
                        try:
                            if self.nn.attenuate!=None:
                                actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p)
                                critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p)
                        except AttributeError:
                            pass
            try:
                param=self.nn.opt(gradient)
            except TypeError:
                param=self.nn.opt(gradient,p)
            lock[0].release()
        elif self.PO==2:
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return 0
            try:
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                except AttributeError:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            lock[0].release()
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[1].acquire()
            if self.stop_func_(lock[1]):
                return 0
            try:
                if self.nn.attenuate!=None:
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)
                    except NameError:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p)
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p)
            except AttributeError:
                pass
            try:
                param=self.nn.opt(gradient)
            except TypeError:
                param=self.nn.opt(gradient,p)
            lock[1].release()
        elif self.PO==3:
            g_lock[ln].acquire()
            if self.stop_func_(g_lock[ln]):
                return 0
            try:
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                except AttributeError:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            g_lock[ln].release()
            if self.priority_flag==True and self.priority_p.value!=-1:
                while True:
                    if p==self.priority_p.value:
                        break
                    else:
                        continue
            lock[0].acquire()
            if self.stop_func_(lock[0]):
                return 0
            try:
                if self.nn.attenuate!=None:
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)
                    except NameError:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p)
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p)
            except AttributeError:
                pass
            try:
                param=self.nn.opt(gradient)
            except TypeError:
                param=self.nn.opt(gradient,p)
            lock[0].release()
        return loss,param
    
    
    def _train(self,p,j,batches,length,lock,g_lock):
        if j==batches-1:
            index1=batches*self.batch
            index2=self.batch-(length-batches*self.batch)
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0)
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0)
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0)
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0)
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0)
            if self.PO==3:
                if len(g_lock)==self.process:
                    ln=p
                else:
                    ln=int(np.random.choice(len(g_lock)))
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock,ln)
            else:
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock)
            self.param[7]=param
            self.loss[p]+=loss
            try:
                bc=self.nn.bc[0]
                bc.assign_add(1)
                self.nn.bc[0]=bc
            except AttributeError:
                pass
        else:
            index1=j*self.batch
            index2=(j+1)*self.batch
            state_batch=self.state_pool[p][index1:index2]
            action_batch=self.action_pool[p][index1:index2]
            next_state_batch=self.next_state_pool[p][index1:index2]
            reward_batch=self.reward_pool[p][index1:index2]
            done_batch=self.done_pool[p][index1:index2]
            if self.PO==3:
                if len(g_lock)==self.process:
                    ln=p
                else:
                    ln=int(np.random.choice(len(g_lock)))
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock,ln)
            else:
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock)
            self.param[7]=param
            self.loss[p]+=loss
            try:
                bc=self.nn.bc[0]
                bc.assign_add(1)
                self.nn.bc[0]=bc
            except AttributeError:
                pass
        return
    
    
    def train_(self,p,lock,g_lock):
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
                try:
                    if self.nn.attenuate!=None:
                        opt_counter=self.nn.opt_counter[0]
                        opt_counter.scatter_update(tf.IndexedSlices(0,p))
                        self.nn.opt_counter[0]=opt_counter
                except AttributeError:
                    pass
                self._train(p,j,batches,length,lock,g_lock)
                if self.priority_flag==True:
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')
                    opt_counter+=1
                try:
                    if self.nn.attenuate!=None:
                        opt_counter=self.nn.opt_counter[0]
                        opt_counter.assign(opt_counter+1)
                        self.nn.opt_counter[0]=opt_counter
                except AttributeError:
                    pass
            if self.PO==1 or self.PO==3:
                lock[1].acquire()
            else:
                lock[2].acquire()
            if self.update_step!=None:
                if self.sc[p]%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
            if self.PO==1 or self.PO==3:
                lock[1].release()
            else:
                lock[2].release()
            self.loss[p]=self.loss[p]/batches
        self.sc[p]+=1
        try:
            ec=self.nn.ec[0]
            ec.assign_add(1)
            self.nn.ec[0]=ec
        except AttributeError:
            pass
        return
    
    
    def train(self,p,episode_count,lock,pool_lock,g_lock=None):
        if self.PO==1 or self.PO==3:
            lock[2].acquire()
        else:
            lock[3].acquire()
        self.state_pool[p]=None
        self.action_pool[p]=None
        self.next_state_pool[p]=None
        self.reward_pool[p]=None
        self.done_pool[p]=None
        self.running_flag.append(1)
        self.process_counter.value+=1
        self.finish_list.append(None)
        try:
            epsilon=self.epsilon[p]
        except:
            epsilon=None
        if self.PO==1 or self.PO==3:
            lock[2].release()
        else:
            lock[3].release()
        for k in range(episode_count):
            s=self.nn.env(p=p,initial=True)
            if type(self.nn.param[0])!=list:
                s=np.array(s,self.nn.param[0].dtype.name)
            else:
                s=np.array(s,self.nn.param[0][0].dtype.name)
            if self.episode_step==None:
                while True:
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)
                    self.reward[p]+=r
                    s=next_s
                    if type(self.done_pool[p])==np.ndarray:
                        self.train_(p,lock,g_lock)
                    if done:
                        if self.PO==1 or self.PO==3:
                            lock[2].acquire()
                        else:
                            lock[3].acquire()
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if self.PO==1 or self.PO==3:
                            lock[2].release()
                        else:
                            lock[3].release()
                        break
            else:
                for l in range(self.episode_step):
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)
                    self.reward[p]+=r
                    s=next_s
                    if type(self.done_pool[p])==np.ndarray:
                        self.train_(p,lock,g_lock)
                    if done:
                        if self.PO==1 or self.PO==3:
                            lock[2].acquire()
                        else:
                            lock[3].acquire()
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if self.PO==1 or self.PO==3:
                            lock[2].release()
                        else:
                            lock[3].release()
                        break
                    if l==self.episode_step-1:
                        if self.PO==1 or self.PO==3:
                            lock[2].acquire()
                        else:
                            lock[3].acquire()
                        self.total_episode.value+=1
                        self.loss_list.append(self.loss[p])
                        if self.PO==1 or self.PO==3:
                            lock[2].release()
                        else:
                            lock[3].release()
            if self.PO==1 or self.PO==3:
                lock[2].acquire()
            else:
                lock[3].acquire()
            self.reward_list.append(self.reward[p])
            self.reward[p]=0
            if self.PO==1 or self.PO==3:
                lock[2].release()
            else:
                lock[3].release()
        self.running_flag[p+1]=0
        if p not in self.finish_list:
            self.finish_list[p]=p
        if self.PO==1 or self.PO==3:
            lock[2].acquire()
        else:
            lock[3].acquire()
        self.process_counter.value-=1
        if self.PO==1 or self.PO==3:
            lock[2].release()
        else:
            lock[3].release()
        del self.state_pool[p]
        del self.action_pool[p]
        del self.next_state_pool[p]
        del self.reward_pool[p]
        del self.done_pool[p]
        return
    
    
    def stop_func(self):
        if self.end():
            self.save(self.total_episode)
            self.save_flag.value=True
            self.stop_flag.value=True
            return True
        return False
    
    
    def stop_func_(self,lock):
        if self.stop==True:
            if self.stop_flag.value==True or self.stop_func():
                lock.release
                return True
    
    
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
            self.file_list.append([filename])
            if len(self.file_list)>self.s+1:
                os.remove(self.file_list[0][0])
                del self.file_list[0]
        pickle.dump(self.nn,output_file)
        pickle.dump(self.param[7],output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(np.array(self.sc),output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.total_episode.value,output_file)
        try:
            pickle.dump(self.nn.ec,output_file)
        except AttributeError:
            pass
        try:
            pickle.dump(self.nn.bc,output_file)
        except AttributeError:
            pass
        output_file.close()
        return
    
    
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.param[7]=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.sc=pickle.load(input_file)
        self.sc=Array('f',self.sc)
        self.update_step=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.total_episode.value=pickle.load(input_file)
        self.nn.ec=pickle.load(input_file)
        self.nn.bc=pickle.load(input_file)
        input_file.close()
        return
