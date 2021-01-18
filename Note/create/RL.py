import tensorflow as tf
import numpy as np


def epsilon_greedy_policy(action,action_p,epsilon):
    action_prob=action_p
    action_prob=action_prob*epsilon/np.sum(action_p)
    best_a=np.argmax(action)
    action_prob[best_a]+=1-epsilon
    return action_prob


def pool(state_pool,action_pool,next_state_pool,reward_pool,state,action,next_state,reward,pool_size):
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