import numpy as np


def pool_t(state_pool,action_pool,next_state_pool,reward_pool,done_pool,sanrd,pool_size,t,index,pool_lock,PN=True):
    if PN==True:
        pool_lock[index].acquire()
        if type(state_pool[index])!=np.ndarray and state_pool[index]==None:
            state_pool[index]=sanrd[0]
            if type(sanrd[1])==int:
                a=np.array(sanrd[1],np.int32)
                action_pool[index]=np.expand_dims(a,axis=0)
            else:
                action_pool[index]=sanrd[1]
            next_state_pool[index]=np.expand_dims(sanrd[2],axis=0)
            reward_pool[index]=np.expand_dims(sanrd[3],axis=0)
            done_pool[index]=np.expand_dims(sanrd[4],axis=0)
        else:
            try:
                state_pool[index]=np.concatenate((state_pool[index],sanrd[0]),0)
                if type(sanrd[1])==int:
                    a=np.array(sanrd[1],np.int32)
                    action_pool[index]=np.concatenate((action_pool[index],np.expand_dims(a,axis=0)),0)
                else:
                    action_pool[index]=np.concatenate((action_pool[index],sanrd[1]),0)
                next_state_pool[index]=np.concatenate((next_state_pool[index],np.expand_dims(sanrd[2],axis=0)),0)
                reward_pool[index]=np.concatenate((reward_pool[index],np.expand_dims(sanrd[3],axis=0)),0)
                done_pool[index]=np.concatenate((done_pool[index],np.expand_dims(sanrd[4],axis=0)),0)
            except:
                pass
        try:
            if type(state_pool[index])==np.ndarray and len(state_pool[index])>pool_size:
                state_pool[index]=state_pool[index][1:]
                action_pool[index]=action_pool[index][1:]
                next_state_pool[index]=next_state_pool[index][1:]
                reward_pool[index]=reward_pool[index][1:]
                done_pool[index]=done_pool[index][1:]
                del state_pool[t]
                del action_pool[t]
                del next_state_pool[t]
                del reward_pool[t]
                del done_pool[t]
        except:
            pass
        pool_lock[index].release()
    else:
        if type(state_pool[t])==np.ndarray and state_pool[t]==None:
            state_pool[t]=sanrd[0]
            if type(sanrd[1])==int:
                a=np.array(sanrd[1],np.int32)
                action_pool[t]=np.expand_dims(a,axis=0)
            else:
                action_pool[t]=sanrd[1]
            next_state_pool[t]=np.expand_dims(sanrd[2],axis=0)
            reward_pool[t]=np.expand_dims(sanrd[3],axis=0)
            done_pool[t]=np.expand_dims(sanrd[4],axis=0)
        else:
            state_pool[t]=np.concatenate((state_pool[t],sanrd[0]),0)
            if type(sanrd[1])==int:
                a=np.array(sanrd[1],np.int32)
                action_pool[t]=np.concatenate((action_pool[t],np.expand_dims(a,axis=0)),0)
            else:
                action_pool[t]=np.concatenate((action_pool[t],a),0)
            next_state_pool[t]=np.concatenate((next_state_pool[t],np.expand_dims(sanrd[2],axis=0)),0)
            reward_pool[t]=np.concatenate((reward_pool[t],np.expand_dims(sanrd[3],axis=0)),0)
            done_pool[t]=np.concatenate((done_pool[t],np.expand_dims(sanrd[4],axis=0)),0)
        if state_pool[t]!=None and len(state_pool[t])>pool_size:
            state_pool[t]=state_pool[t][1:]
            action_pool[t]=action_pool[t][1:]
            next_state_pool[t]=next_state_pool[t][1:]
            reward_pool[t]=reward_pool[t][1:]
            done_pool[t]=done_pool[t][1:]
    return


def pool(state_pool,action_pool,next_state_pool,reward_pool,done_pool,sanrd,pool_size):
    if type(state_pool)!=np.ndarray and state_pool==None:
        state_pool=sanrd[0]
        if type(sanrd[1])==int:
            a=np.array(sanrd[1],np.int32)
            action_pool=np.expand_dims(a,axis=0)
        else:
            action_pool=sanrd[1]
        next_state_pool=np.expand_dims(sanrd[2],axis=0)
        reward_pool=np.expand_dims(sanrd[3],axis=0)
        done_pool=np.expand_dims(sanrd[4],axis=0)
    else:
        state_pool=np.concatenate((state_pool,sanrd[0]),0)
        if type(sanrd[1])==int:
            a=np.array(sanrd[1],np.int32)
            action_pool=np.concatenate((action_pool,np.expand_dims(a,axis=0)),0)
        else:
            action_pool=np.concatenate((action_pool,sanrd[1]),0)
        next_state_pool=np.concatenate((next_state_pool,np.expand_dims(sanrd[2],axis=0)),0)
        reward_pool=np.concatenate((reward_pool,np.expand_dims(sanrd[3],axis=0)),0)
        done_pool=np.concatenate((done_pool,np.expand_dims(sanrd[4],axis=0)),0)
    if len(state_pool)>pool_size:
        state_pool=state_pool[1:]
        action_pool=action_pool[1:]
        next_state_pool=next_state_pool[1:]
        reward_pool=reward_pool[1:]
        done_pool=done_pool[1:]
    return
