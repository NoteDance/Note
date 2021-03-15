import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class NoisyNet:
    def __init__(self,value_net,state,state_name,action_name,exploration_space,DUELING=None,save_episode=True):
        self.value_net=value_net
        self.value_p=None
        self.target_p=None
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.action_len=len(self.action_name)
        self.DUELING=DUELING
        self.epsilon=None
        self.discount=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.optimizer=None
        self.lr=None
        self.end_loss=None
        self.save_episode=save_episode
        self.loss_list=[]
        self.opt_flag==False
        self.a=0
        self.epi_num=0
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def init(self,dtype=np.int32):
        self.t3=time.time()
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
        self.index=np.arange(self.batch,dtype=np.int8)
        self.t4=time.time()
        return
    
    
    def set_up(self,value_p=None,target_p=None,epsilon=None,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,optimizer=None,lr=None,end_loss=None,init=False):
        if value_p!=None:
            self.value_p=value_p
            self.target_p=target_p
        if epsilon!=None:
            self.epsilon=epsilon
        if discount!=None:
            self.discount=discount
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
            self.index=np.arange(self.batch,dtype=np.int8)
        if update_step!=None:
            self.update_step=update_step
        if optimizer!=None:
            self.optimizer=optimizer
        if lr!=None:
            self.lr=lr
            self.optimizer.lr=lr
        if end_loss!=None:
            self.end_loss=end_loss
        if init==True:
            self.episode=[]
            self.state_pool=None
            self.action_pool=None
            self.next_state_pool=None
            self.reward_pool=None
            self.loss_list=[]
            self.a=0
            self.epi_num=0
            self.episode_num=0
            self.total_episode=0
            self.time=0
            self.total_time=0
        return
    
    
    def noise(self,x):
        return tf.math.sign(x)*tf.math.sqrt(tf.math.abs(x))
    
    
    def Gaussian_noise(self,value_p):
        noise=[]
        for i in range(len(value_p)):
            noise_row=self.noise(tf.random.normal([value_p[i].shape[0],1]),dtype=value_p[i].dtype)
            noise_column=self.noise(tf.random.normal([value_p[i].shape[1],1]),dtype=value_p[i].dtype)
            noise_bias=self.noise(tf.random.normal([value_p[i].shape[1],1]),dtype=value_p[i].dtype)
            noise.append([noise_row,noise_column,noise_bias])
        return noise
    
    
    def update_parameter(self):
        for i in range(len(self.value_p)):
            self.target_p[i]=self.value_p[i].copy()
        return
    
    
    def loss(self,s,a,next_s,r):
        if self.DUELING==False:
            noisy1=self.noisy_variable(self.target_p[0])
            noisy2=self.noisy_variable(self.value_p[0])
            if len(self.state_pool)<self.batch:
                return tf.reduce_mean(((r+self.discount*tf.reduce_max(self.value_net(next_s,self.target_p,noisy1),axis=-1))-self.value_net(s,self.value_p,noisy2)[np.arange(len(a)),a])**2)
            else:
                return tf.reduce_mean(((r+self.discount*tf.reduce_max(self.value_net(next_s,self.target_p,noisy1),axis=-1))-self.value_net(s,self.value_p,noisy2)[self.index,a])**2)
        else:
            noisy1=self.noisy_variable(self.target_p[0])
            noisy2=self.noisy_variable(self.value_p[0])
            value1,action1=self.value_net(next_s,self.target_p,noisy1)
            value2,action2=self.value_net(s,self.value_p,noisy2)
            action1=action1-tf.expand_dims(tf.reduce_sum(action1,axis=-1)/self.action,axis=-1)
            action2=action2-tf.expand_dims(tf.reduce_sum(action2,axis=-1)/self.action,axis=-1)
            Q1=value1+action1
            Q2=value2+action2
            if len(self.state_pool)<self.batch:
                return tf.reduce_mean(((r+self.discount*tf.reduce_max(Q1,axis=-1))-Q2[np.arange(len(a)),a])**2)
            else:
                return tf.reduce_mean(((r+self.discount*tf.reduce_max(Q1,axis=-1))-Q2[self.index,a])**2)
    
    
    def learn1(self,parameter,index):
        if len(self.state_pool)<self.batch:
            with tf.GradientTape() as tape:
                loss=self.loss(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool)
            gradient=tape.gradient(loss,parameter)
            if self.opt_flag==True:
                self.optimizer(gradient,parameter)
            else:
                self.optimizer.apply_gradients(zip(gradient,parameter))
            if self.a%self.update_step==0:
                self.value_p[0]=parameter[:index]
                self.update_parameter()
                self.value_p[0].extend(self.value_p[1])
                parameter=self.value_p[0]
        else:
            loss=0
            self.batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool)).shuffle(len(self.state_pool)).batch(self.batch)
            for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
                with tf.GradientTape() as tape:
                    batch_loss=self.loss(state_batch,action_batch,next_state_batch,reward_batch)
                gradient=tape.gradient(batch_loss,parameter)
                if self.opt_flag==True:
                    self.optimizer(gradient,parameter)
                else:
                    self.optimizer.apply_gradients(zip(gradient,parameter))
                loss+=batch_loss
            if self.a%self.update_step==0:
                self.value_p[0]=parameter[:index]
                self.update_parameter()
                self.value_p[0].extend(self.value_p[1])
                parameter=self.value_p[0]
            if len(self.state_pool)%self.batch!=0:
                loss=loss.numpy()/self.batches+1
            elif len(self.state_pool)<self.batch:
                loss=loss.numpy()
            else:
                loss=loss.numpy()/self.batches
        return loss
    
    
    def learn2(self,parameter,index):
        episode=[]
        s=int(np.random.uniform(0,len(self.state_name)))
        if self.episode_step==None:
            while True:
                t1=time.time()
                self.a+=1
                noisy=self.noisy_variable(self.value_p[0])
                value=self.value_net(self.state_name[s],self.value_p,noisy)
                a=np.argmax(value)
                next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                if self.state_pool==None:
                    self.state_pool=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                    self.action_pool=tf.expand_dims(a,axis=0)
                    self.next_state_pool=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool=tf.expand_dims(r,axis=0)
                else:
                    self.state_pool=tf.concat([self.state_pool,tf.expand_dims(self.state[self.state_name[s]],axis=0)])
                    self.action_pool=tf.concat([self.action_pool,tf.expand_dims(a,axis=0)])
                    self.next_state_pool=tf.concat([self.next_state_pool,tf.expand_dims(self.state[self.state_name[next_s]],axis=0)])
                    self.reward_pool=tf.concat([self.reward_pool,tf.expand_dims(r,axis=0)])
                if len(self.state_pool)>self.pool_size:
                    self.state_pool=self.state_pool[1:]
                    self.action_pool=self.action_pool[1:]
                    self.next_state_pool=self.next_state_pool[1:]
                    self.reward_pool=self.reward_pool[1:]
                if end:
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                elif self.save_episode==True:
                    episode.append([self.state_name[s],self.self.action_name[a],r])
                s=next_s
                loss=self.learn1(parameter,index)
                t2=time.time()
                self.time+=(t2-t1)
        else:
            for _ in range(self.episode_step):
                t1=time.time()
                self.a+=1
                noisy=self.noisy_variable(self.value_p[0])
                value=self.value_net(self.state_name[s],self.value_p,noisy)
                a=np.argmax(value)
                next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                if self.state_pool==None:
                    self.state_pool=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                    self.action_pool=tf.expand_dims(a,axis=0)
                    self.next_state_pool=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool=tf.expand_dims(r,axis=0)
                else:
                    self.state_pool=tf.concat([self.state_pool,tf.expand_dims(self.state[self.state_name[s]],axis=0)])
                    self.action_pool=tf.concat([self.action_pool,tf.expand_dims(a,axis=0)])
                    self.next_state_pool=tf.concat([self.next_state_pool,tf.expand_dims(self.state[self.state_name[next_s]],axis=0)])
                    self.reward_pool=tf.concat([self.reward_pool,tf.expand_dims(r,axis=0)])
                if len(self.state_pool)>self.pool_size:
                    self.state_pool=self.state_pool[1:]
                    self.action_pool=self.action_pool[1:]
                    self.next_state_pool=self.next_state_pool[1:]
                    self.reward_pool=self.reward_pool[1:]
                if end:
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.action_name[a],r,end])
                    break
                elif self.save_episode==True:
                    episode.append([self.state_name[s],self.self.action_name[a],r])
                s=next_s
                loss=self.learn1(parameter,index)
                t2=time.time()
                self.time+=(t2-t1)
        return loss,episode
            
    
    def learn(self,episode_num,path=None,one=True):
        index=len(self.value_p[0])
        self.value_p[0].extend(self.value_p[1])
        parameter=self.value_p[0]
        if self.lr!=None:
            self.optimizer.lr=self.lr
        if episode_num!=None:
            for i in range(episode_num):
                loss,episode=self.learn2(parameter,index)
                self.loss_list.append(loss)
                if episode_num%10!=0:
                    d=episode_num-episode_num%10
                    d=int(d/10)
                else:
                    d=episode_num/10
                if d==0:
                    d=1
                if i%d==0:
                    print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if path!=None and i%episode_num*2==0:
                        self.save(path,i,one)
                self.epi_num+=1
                self.total_episode+=1
                if self.save_episode==True:
                    self.episode.append(episode)
                if self.end_loss!=None and loss<=self.end_loss:
                    break
        else:
            while True:
                loss,episode=self.learn2(parameter,index)
                self.loss_list.append(loss)
                if episode_num%10!=0:
                    d=episode_num-episode_num%10
                    d=int(d/10)
                else:
                    d=episode_num/10
                if d==0:
                    d=1
                if i%d==0:
                    print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if path!=None and i%episode_num*2==0:
                        self.save(path,i,one)
                self.epi_num+=1
                self.total_episode+=1
                if self.save_episode==True:
                    self.episode.append(episode)
                if self.end_loss!=None and loss<=self.end_loss:
                    break
        if path!=None:
            self.save(path)
        self.value_p[0]=parameter[:index]
        if self.time<0.5:
            self.time=int(self.time+(self.t4-self.t3))
        else:
            self.time=int(self.time+(self.t4-self.t3))+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(loss))
        print('time:{0}s'.format(self.time))
        return
    
    
    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_episode),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump(self.value_p,parameter_file)
        parameter_file.close()
        return
    
    
    def save_e(self,path):
        episode_file=open(path+'.dat','wb')
        pickle.dump(self.episode,episode_file)
        episode_file.close()
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'\save.dat','wb')
            path=path+'\save.dat'
            index=path.rfind('\\')
            episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
        else:
            output_file=open(path+'\save-{0}.dat'.format(i+1),'wb')
            path=path+'\save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            episode_file=open(path.replace(path[index+1:],'episode-{0}.dat'.format(i+1)),'wb')
        self.episode_num=self.epi_num
        pickle.dump(self.episode,episode_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.index,output_file)
        pickle.dump(self.DUELING,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.opt_flag,output_file)
        pickle.dump(self.a,output_file)
        pickle.dump(self.episode_num,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        episode_file.close()
        return
    
    
    def restore(self,s_path,e_path):
        input_file=open(s_path,'rb')
        episode_file=open(e_path,'rb')
        self.episode=pickle.load(episode_file)
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.index=pickle.load(input_file)
        self.DUELING=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.opt_flag=pickle.load(input_file)
        self.a=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        episode_file.close()
        return
