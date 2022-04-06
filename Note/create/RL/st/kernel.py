import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class kernel:
    def __init__(self,nn=None,state=None,state_name=None,action_name=None,exploration_space=None,pr=None,save_episode=True):
        if nn!=None:
            self.nn=nn
            self.opt=nn.opt
            try:
                if self.nn.km==0:
                    self.nn.km=1
            except AttributeError:
                pass
        self.ol=None
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.exploration_space=exploration_space
        self.explore=None
        self.pr=pr
        self.epsilon=None
        self.discount=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.rp=None
        self.alpha=None
        self.beta=None
        self.end_loss=None
        self.save_episode=save_episode
        self.loss_list=[]
        self.a=0
        self.d=None
        self.e=None
        self.epi_num=0
        self.episode_num=0
        self.total_episode=0
        self.total_e=0
        self.time=0
        self.total_time=0
    
    
    def init(self,dtype=np.int32):
        self.action_len=len(self.action_name)
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            if self.epsilon!=None:
                self.action_one=np.ones(len(self.action_name),dtype=dtype)
        return
    
    
    def set_up(self,param=None,epsilon=None,discount=None,episode_step=None,pool_size=None,batch=None,update_step=None,rp=None,alpha=None,beta=None,end_loss=None):
        if param!=None:
            self.nn.param=param
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
        if update_step!=None:
            self.update_step=update_step
        if rp!=None:
            self.rp=rp
        if alpha!=None:
            self.alpha=alpha
        if beta!=None:
            self.beta=beta
        if end_loss!=None:
            self.end_loss=end_loss
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
        self.total_e=0
        self.time=0
        self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_a=np.argmax(self.nn.nn(self.state[self.state_name[s]]))
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def _epsilon_greedy_policy(self,a,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_a=np.argmax(a)
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def learn1(self,episode_num,i):
        if self.end_loss!=None:
            self.param=self.nn.param
        if len(self.state_pool)<self.batch:
            with tf.GradientTape() as tape:
                if type(self.nn.nn)!=list:
                    loss=self.nn.loss(self.nn.nn,self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool)				
                elif len(self.nn.param)==4:
                    value=self.nn.nn[0](self.state_pool,p=0)
                    TD=tf.reduce_mean((self.reward_pool+self.discount*self.nn.nn[0](self.next_state_pool,p=2)-value)**2)
                else:
                    value=self.nn.nn[0](self.state_pool)
                    TD=tf.reduce_mean((self.reward_pool+self.discount*self.nn.nn[0](self.next_state_pool)-value)**2)
            if type(self.nn.nn)!=list:
                gradient=tape.gradient(loss,self.nn.param[0])
                self.opt(gradient,self.nn.param[0])
            elif len(self.nn.param)==4:
                value_gradient=tape.gradient(TD,self.nn.param[0])
                actor_gradient=tape.gradient(value,self.action_pool)*tape.gradient(self.action_pool,self.nn.param[1])
                self.opt(value_gradient,actor_gradient,self.nn.param)
            else:
                value_gradient=tape.gradient(TD,self.nn.param[0])				
                actor_gradient=TD*tape.gradient(tf.math.log(self.action_pool),self.nn.param[1])
                self.opt(value_gradient,actor_gradient,self.nn.param)
            if self.update_step!=None:
                if self.a%self.update_step==0:
                    self.nn.update_param(self.nn.param)
            else:
                self.nn.update_param(self.nn.param)
            loss=0
        else:
            loss=0
            self.loss=0
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if self.pr!=None:
                for j in range(batches):
                    state_batch,action_batch,next_state_batch,reward_batch=self.pr(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.pool_size,self.batch,self.rp,self.alpha,self.beta)
                    with tf.GradientTape() as tape:
                        if type(self.nn.nn)!=list:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        elif len(self.nn.param)==4:
                            value=self.nn.nn[0](self.state_pool,p=0)
                            TD=tf.reduce_mean((self.reward_pool+self.discount*self.nn.nn[0](self.next_state_pool,p=2)-value)**2)
                        else:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                    if type(self.nn.nn)!=list:
                        gradient=tape.gradient(batch_loss,self.nn.param[0])
                        self.opt(gradient,self.nn.param[0])
                        if j>=1:
                            loss+=batch_loss
                        if i==episode_num-1:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                            self.loss+=batch_loss
                    elif len(self.nn.param)==4:
                        value_gradient=tape.gradient(TD,self.nn.param[0])
                        actor_gradient=tape.gradient(value,self.action_pool)*tape.gradient(self.action_pool,self.nn.param[1])
                        self.opt(value_gradient,actor_gradient,self.nn.param)
                        if j>=1:
                            loss+=TD
                        if i==episode_num-1:
                            value=self.nn.nn[0](state_batch,p=0)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                            self.loss+=TD
                    else:
                        value_gradient=tape.gradient(TD,self.nn.param[0])
                        actor_gradient=TD*tape.gradient(tf.math.log(action_batch),self.nn.param[1])
                        self.opt(value_gradient,actor_gradient,self.nn.param)
                        if j>=1:
                            loss+=TD
                        if i==episode_num-1:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                            self.loss+=TD
                    try:
                        self.nn.bc=j
                    except AttributeError:
                        pass
                if len(self.state_pool)%self.batch!=0:
                    state_batch,action_batch,next_state_batch,reward_batch=self.pr(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.pool_size,self.batch,self.rp,self.alpha,self.beta)
                    with tf.GradientTape() as tape:
                        if type(self.nn.nn)!=list:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        elif len(self.nn.param)==4:
                            value=self.nn.nn[0](self.state_pool,p=0)
                            TD=tf.reduce_mean((self.reward_pool+self.discount*self.nn.nn[0](self.next_state_pool,p=2)-value)**2)
                        else:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                    if type(self.nn.nn)!=list:
                        gradient=tape.gradient(batch_loss,self.nn.param[0])
                        self.opt(gradient,self.nn.param[0])
                        loss+=batch_loss
                        if i==episode_num-1:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                            self.loss+=batch_loss
                    elif len(self.nn.param)==4:
                        value_gradient=tape.gradient(TD,self.nn.param[0])
                        actor_gradient=tape.gradient(value,self.action_pool)*tape.gradient(self.action_pool,self.nn.param[1])
                        self.opt(value_gradient,actor_gradient,self.nn.param)
                        if j>=1:
                            loss+=TD
                        if i==episode_num-1:
                            value=self.nn.nn[0](state_batch,p=0)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                            self.loss+=TD
                    else:
                        value_gradient=tape.gradient(TD,self.nn.param[0])
                        actor_gradient=TD*tape.gradient(tf.math.log(action_batch),self.nn.param[1])
                        self.opt(value_gradient,actor_gradient,self.nn.param)
                        loss+=TD
                        if i==episode_num-1:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                            self.loss+=TD
                    try:
                        self.nn.bc+=1
                    except AttributeError:
                        pass
            else:
                j=0
                train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                try:
                    self.nn.bc=0
                except AttributeError:
                    pass
                for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
                    with tf.GradientTape() as tape:
                        if type(self.nn.nn)!=list:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        elif len(self.nn.param)==4:
                            value=self.nn.nn[0](self.state_pool,p=0)
                            TD=tf.reduce_mean((self.reward_pool+self.discount*self.nn.nn[0](self.next_state_pool,p=2)-value)**2)
                        else:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                    if type(self.nn.nn)!=list:
                        gradient=tape.gradient(batch_loss,self.nn.param[0])
                        self.opt(gradient,self.nn.param[0])
                        if j>=1:
                            loss+=batch_loss
                        if i==episode_num-1:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                            self.loss+=batch_loss
                    elif len(self.nn.param)==4:
                        value_gradient=tape.gradient(TD,self.nn.param[0])
                        actor_gradient=tape.gradient(value,self.action_pool)*tape.gradient(self.action_pool,self.nn.param[1])
                        self.opt(value_gradient,actor_gradient,self.nn.param)
                        if j>=1:
                            loss+=TD
                        if i==episode_num-1:
                            value=self.nn.nn[0](state_batch,p=0)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                            self.loss+=TD
                    else:
                        value_gradient=tape.gradient(TD,self.nn.param[0])
                        actor_gradient=TD*tape.gradient(tf.math.log(action_batch),self.nn.param[1])
                        self.opt(value_gradient,actor_gradient,self.nn.param)
                        if j>=1:
                            loss+=TD
                        if i==episode_num-1:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                            self.loss+=TD
                    j+=1
                    try:
                        self.nn.bc+=1
                    except AttributeError:
                        pass
            if self.update_step!=None:
                if self.a%self.update_step==0:
                    self.nn.update_param(self.nn.param)
            else:
                self.nn.update_param(self.nn.param)
            if len(self.state_pool)<self.batch:
                loss=loss.numpy()
            else:
                loss=loss.numpy()/batches
                if i==episode_num-1:
                    self.loss=self.loss.numpy()/batches
        return loss
    
    
    def learn2(self,episode_num,i):
        episode=[]
        if self.exploration_space==None:
            s=self.exploration.explore(init=True)
        else:
            s=int(np.random.uniform(0,len(self.state_name)))
        if self.episode_step==None:
            while True:
                t1=time.time()
                self.a+=1
                if type(self.nn.nn)!=list:
                    if self.explore==None:
                        action_prob=self.epsilon_greedy_policy(s,self.action_one)
                        a=np.random.choice(self.action,p=action_prob)
                        next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    else:
                        if self.exploration_space==None:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                            next_s,r,end=self.exploration.explore(self.action_name[a])
                        else:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                            next_s,r,end=self.exploration.explore(self.state_name[s],self.action_name[a],self.exploration_space[self.state_name[s]][self.action_name[a]])
                else:
                    if self.explore==None:
                        if len(self.nn.param)==4:
                            a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+tf.random.normal([1])).numpy()
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                        if type(self.exploration_space)==dict:
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                        else:
                            next_s,r,end=self.exploration_space(self.state_name[s],a)
                    else:
                        if self.exploration_space==None:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                            if type(self.exploration_space)==dict:
                                a=self._epsilon_greedy_policy(a,self.action_one)
                                next_s,r,end=self.explore(self.action_name[a])
                            else:
                                next_s,r,end=self.explore(a)
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.exploration.explore(self.state_name[s],self.action_name[a],self.exploration_space[self.state_name[s]][self.action_name[a]])
                if self.state_pool==None:
                    if self.exploration_space==None:
                        self.state_pool=tf.expand_dims(s,axis=0)
                        self.action_pool=tf.expand_dims(a,axis=0)
                        self.next_state_pool=tf.expand_dims(next_s,axis=0)
                        self.reward_pool=tf.expand_dims(r,axis=0)
                    else:
                        self.state_pool=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                        self.action_pool=tf.expand_dims(a,axis=0)
                        self.next_state_pool=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                        self.reward_pool=tf.expand_dims(r,axis=0)
                else:
                    if self.exploration_space==None:
                        self.state_pool=tf.concat([self.state_pool,tf.expand_dims(s,axis=0)])
                        self.action_pool=tf.concat([self.action_pool,tf.expand_dims(a,axis=0)])
                        self.next_state_pool=tf.concat([self.next_state_pool,tf.expand_dims(next_s,axis=0)])
                        self.reward_pool=tf.concat([self.reward_pool,tf.expand_dims(r,axis=0)])
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
                        if self.exploration_space==None:
                            episode.append([s,a,next_s,r,end])
                        else:
                            episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r,end])
                    break
                elif self.save_episode==True:
                    if self.exploration_space==None:
                        episode.append([s,a,next_s,r])
                    else:
                        episode.append([self.state_name[s],self.self.action_name[a],self.state_name[next_s],r])
                s=next_s
                loss=self.learn1(episode_num,i)
                t2=time.time()
                self.time+=(t2-t1)
        else:
            for _ in range(self.episode_step):
                t1=time.time()
                self.a+=1
                if type(self.nn.nn)!=list:
                    if self.explore==None:
                        action_prob=self.epsilon_greedy_policy(s,self.action_one)
                        a=np.random.choice(self.action,p=action_prob)
                        next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                    else:
                        if self.exploration_space==None:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                            next_s,r,end=self.explore(self.action_name[a])
                        else:
                            action_prob=self.epsilon_greedy_policy(s,self.action_one)
                            a=np.random.choice(self.action,p=action_prob)
                            next_s,r,end=self.explore(self.state_name[s],self.action_name[a],self.exploration_space[self.state_name[s]][self.action_name[a]])
                else:
                    if self.explore==None:
                        if len(self.nn.param)==4:
                            a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+tf.random.normal([1])).numpy()
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                        if len(a.shape)>0:
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.exploration_space[self.state_name[s]][self.action_name[a]]
                        else:
                            next_s,r,end=self.exploration_space(self.state_name[s],a)
                    else:
                        if self.exploration_space==None:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                            if len(a.shape)>0:
                                a=self._epsilon_greedy_policy(a,self.action_one)
                                next_s,r,end=self.explore(self.action_name[a])
                            else:
                                next_s,r,end=self.explore(a)
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]])
                            a=self._epsilon_greedy_policy(a,self.action_one)
                            next_s,r,end=self.exploration.explore(self.state_name[s],self.action_name[a],self.exploration_space[self.state_name[s]][self.action_name[a]])
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
                        if self.exploration_space==None:
                            episode.append([s,a,next_s,r,end])
                        else:
                            if len(a.shape)>0:
                                episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r,end]
                            else:
                                episode=[self.state_name[s],a,self.state_name[next_s],r,end] 
                    break
                elif self.save_episode==True:
                    if self.exploration_space==None:
                        episode.append([s,a,next_s,r])
                    else:
                        if len(a.shape)>0:
                            episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
                        else:
                            episode=[self.state_name[s],a,self.state_name[next_s],r]
                s=next_s
                loss=self.learn1(episode_num,i)
                t2=time.time()
                self.time+=(t2-t1)
        return loss,episode
    
    
    def learn(self,episode_num,path=None,one=True,p=None,s=None):
        if p==None and s==None:
            self.p=9
            self.s=2
        elif p!=None:
            self.p=p-1
            self.s=2
        elif s!=None:
            self.p=9
            self.s=s
        else:
            self.p=p-1
            self.s=s
        loss=0
        if episode_num!=None:
            for i in range(episode_num):
                loss,episode=self.learn2(episode_num,i)
                self.loss_list.append(loss)
                if i==episode_num-1:
                    self.loss_list.append(self.loss)
                if episode_num%10!=0:
                    d=episode_num-episode_num%self.p
                    d=int(d/self.p)
                else:
                    d=episode_num/(self.p+1)
                    d=int(d)
                if d==0:
                    d=1
                e=d*self.s
                if i%d==0:
                    print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if path!=None and i%e==0:
                        self.save(path,i,one)
                self.epi_num+=1
                self.total_episode+=1
                if self.save_episode==True:
                    self.episode.append(episode)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                if self.end_loss!=None and loss<=self.end_loss:
                    self.nn.param=self.param
                    self.param=None
                    break
        elif self.ol==None:
            i=0
            while True:
                loss,episode=self.learn2(episode_num,i)
                self.loss_list.append(loss)
                if i==episode_num-1:
                    self.loss_list.append(self.loss)
                i+=1
                if episode_num%10!=0:
                    d=episode_num-episode_num%self.p
                    d=int(d/self.p)
                else:
                    d=episode_num/(self.p+1)
                    d=int(d)
                if d==0:
                    d=1
                e=d*self.s
                if i%d==0:
                    print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if path!=None and i%e==0:
                        self.save(path,i,one)
                self.epi_num+=1
                self.total_e+=1
                if self.save_episode==True:
                    self.episode.append(episode)
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
                if self.end_loss!=None and loss<=self.end_loss:
                    self.nn.param=self.param
                    self.param=None
                    break
        else:
            while True:
                data=self.ol()
                if data=='end':
                    if type(self.nn.nn)!=list:
                        loss=self.nn.loss(self.nn.nn,data[0],data[1],data[2],data[3])
                    else:
                        value=self.nn.nn[0](data[0],param=0)
                        TD=tf.reduce_mean((data[3]+self.discount*self.nn.nn[0](data[2],param=1)-value)**2)
                        loss=TD
                    if len(self.loss_list)==0:
                        self.loss_list.append(loss.numpy())
                    else:
                        self.loss_list[0]=loss.numpy()
                    if path!=None:
                        self.save(path)
                    return
                with tf.GradientTape() as tape:
                    if type(self.nn.nn)!=list:
                        loss=self.nn.loss(self.nn.nn,data[0],data[1],data[2],data[3])				
                    else:  
                        value=self.nn.nn[0](data[0],param=0)
                        TD=tf.reduce_mean((data[3]+self.discount*self.nn.nn[0](data[2],param=1)-value)**2)
                if type(self.nn.nn)!=list:
                    gradient=tape.gradient(loss,self.nn.param[0])
                    self.opt.opt(gradient,self.nn.param[0])
                else:
                    value_gradient=tape.gradient(TD,self.nn.param[0])				
                    actor_gradient=TD*tape.gradient(tf.math.log(data[1]),self.nn.param[2])
                    loss=TD
                    self.opt.opt(value_gradient,actor_gradient,self.nn.param)
                self.total_e+=1
                if self.update_step!=None:
                    if self.a%self.update_step==0:
                        self.nn.update_param(self.nn.param)
                else:
                    self.nn.update_param(self.nn.param)
                if len(self.loss_list)==0:
                    self.loss_list.append(loss.numpy())
                else:
                    self.loss_list[0]=loss.numpy()
                try:
                    self.nn.ec+=1
                except AttributeError:
                    pass
        if path!=None:
            self.save(path)
        if self.time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(loss))
        print('time:{0}s'.format(self.time))
        try:
            if self.nn.km==1:
                self.nn.km=0
        except AttributeError:
            pass
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
        pickle.dump(self.nn.param,parameter_file)
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
            parameter_file=open(path.replace(path[index+1:],'parameter.dat'),'wb')
            if self.save_episode==True:
                episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
        else:
            output_file=open(path+'\save-{0}.dat'.format(i+1),'wb')
            path=path+'\save-{0}.dat'.format(i+1)
            index=path.rfind('\\')
            parameter_file=open(path.replace(path[index+1:],'parameter-{0}.dat'.format(i+1)),'wb')
            if self.save_episode==True:
                episode_file=open(path.replace(path[index+1:],'episode-{0}.dat'.format(i+1)),'wb')
                pickle.dump(self.episode,episode_file)
                episode_file.close()
        self.episode_num=self.epi_num
        pickle.dump(self.nn.param,parameter_file)
        self.nn.param=None
        pickle.dump(self.nn,output_file)
        pickle.dump(self.ol,output_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.exploration,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.rp,output_file)
        pickle.dump(self.alpha,output_file)
        pickle.dump(self.beta,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.a,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.s,output_file)
        pickle.dump(self.episode_num,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_e,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,p_path,e_path=None):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        if e_path!=None:
            episode_file=open(e_path,'rb')
            self.episode=pickle.load(episode_file)
            episode_file.close()
        param=pickle.load(parameter_file)
        self.nn=pickle.load(input_file)
        self.nn.param=param
        param=None
        try:
            if self.nn.km==0:
                self.nn.km=1
        except AttributeError:
            pass
        self.opt=self.nn.opt
        self.ol=pickle.load(input_file)
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.exploration=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.rp=pickle.load(input_file)
        self.alpha=pickle.load(input_file)
        self.beta=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.a=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.s=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_e=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
