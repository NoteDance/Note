import tensorflow as tf
import numpy as np


def epsilon_greedy_policy(action,value,action_one,epsilon):
    action_prob=action_one
    action_prob=action_prob*epsilon/len(action_one)
    best_a=np.argmax(value)
    action_prob[best_a]+=1-epsilon
    return np.random.choice(action,p=action_prob)


def pool(state,action,next_state,reward,pool_size,state_pool=None,action_pool=None,next_state_pool=None,reward_pool=None):
   if state_pool==None:
       state_pool=tf.expand_dims(state,axis=0)
       action_pool=tf.expand_dims(action,axis=0)
       next_state_pool=tf.expand_dims(next_state,axis=0)
       reward_pool=tf.expand_dims(reward,axis=0)
   else:
       state_pool=tf.concatenate(state_pool,tf.expand_dims(state,axis=0))
       action_pool=tf.concatenate(action_pool,tf.expand_dims(action,axis=0))
       next_state_pool=tf.concatenate(next_state_pool,tf.expand_dims(next_state,axis=0))
       reward_pool=tf.concatenate(reward_pool,tf.expand_dims(reward,axis=0))
   if len(state_pool)>pool_size:
       state_pool=state_pool[1:]
       action_pool=action_pool[1:]
       next_state_pool=next_state_pool[1:]
       reward_pool=reward_pool[1:]
   return state_pool,action_pool,next_state_pool,reward_pool


def _noise(x):
    return tf.math.sign(x)*tf.math.sqrt(tf.math.abs(x))


def Gaussian_noise(value_p,dtype=tf.float32):
    noise=[]
    for i in range(len(value_p)):
        noise_row=_noise(tf.random.normal([value_p[i].shape[0],1]),dtype=dtype)
        noise_column=_noise(tf.random.normal([value_p[i].shape[1],1]),dtype=dtype)
        noise_bias=_noise(tf.random.normal([value_p[i].shape[1],1]),dtype=dtype)
        noise.append([noise_row,noise_column,noise_bias])
    return noise


def pr(state_pool,t,batch,K,N,alpha,beta,error,p=None):
    if p==None:
        p=tf.ones([len(state_pool)],dtype=tf.float32)
        prob=p**alpha/tf.reduce_sum(p**alpha)
        w=(N*prob)**-beta
        w=w/tf.reduce_max(w)
    else:
        p=tf.concat([p,tf.ones([len(state_pool)-len(p)])*tf.reduce_max(p)])
        prob=p**alpha/tf.reduce_sum(p**alpha)
        w=(N*prob)**-beta
        w=w/tf.reduce_max(w)
    if t%K==0:
        index=np.random.choice(np.arange(len(state_pool),dtype=np.int8),size=[len(state_pool)],p=prob)
        delta=error(state_pool[index])
        p=delta+10**-7
    return w*delta,p,delta
