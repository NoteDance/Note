import numpy as np


def epsilon_greedy_policy(action_num,value,action_one,epsilon):
    action_prob=action_one
    action_prob=action_prob*epsilon/len(action_one)
    best_a=np.argmax(value)
    action_prob[best_a]+=1-epsilon
    return np.random.choice(action_num,p=action_prob)
