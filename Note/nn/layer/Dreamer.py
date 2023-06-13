import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.dense import dense
from Note.nn.LSTMCell import LSTMCell


class Dreamer:
    def __init__(self,state_size,action_size,reward_size,horizon,kernel_size=3,filters=3):
        # define the state size, action size, reward size and horizon
        self.state_size=state_size
        self.action_size=action_size
        self.reward_size=reward_size
        self.horizon=horizon
        # define the kernel size and filters for the convolutional network
        self.kernel_size=kernel_size
        self.filters=filters
        # define the world model as a recurrent network with LSTM cells
        self.world_model=LSTMCell((state_size+action_size,state_size))
        # define the observation model as a convolutional network
        self.observation_model=dense((state_size,kernel_size*kernel_size*filters),activation='relu')
        # define the observation kernel for encoding observations into states
        self.observation_kernel=initializer((kernel_size,kernel_size,3,state_size))
        # define the reconstruction kernel for decoding states into observations
        self.reconstruction_kernel=initializer((kernel_size,kernel_size,3,filters))
        # define the reward model as a dense network
        self.reward_model=dense((state_size,reward_size))
        # define the value model as a dense network
        self.value_model=dense((state_size,1))
        # define the policy model as a dense network
        self.policy_model=dense((state_size,action_size))
        self.param=[self.world_model.param,self.observation_model.param,self.observation_kernel,self.reconstruction_kernel,self.reward_model.param,self.value_model.param,self.policy_model.param]
    
    
    def output(self,data,strides=1,padding='SAME'):
        # data should be a tuple of (observations, actions, rewards)
        observations,actions,rewards=data
        # encode the observations into latent states using conv2d and reshape
        states=tf.nn.conv2d(observations,self.kernel1,strides=strides,padding=padding)
        states=tf.reshape(states,(-1,self.state_size))
        # concatenate the states and actions
        data=tf.concat([states,actions],axis=-1)
        # predict the next states using the world model
        next_states=[]
        state=tf.zeros((data.shape[0],self.state_size))
        for i in range(data.shape[1]):
            output,state=self.world_model.output(data[:,i],state)
            next_states.append(output)
        next_states=tf.stack(next_states,axis=1)
        # predict the next observations using the observation model and conv2d_transpose
        next_observations=[]
        for i in range(next_states.shape[1]):
            output=self.observation_model.output(next_states[:,i])
            output=tf.reshape(output,(-1,self.kernel_size,self.kernel_size,self.filters))
            output=tf.nn.conv2d_transpose(output,self.kernel2,strides=strides,
                                            output_shape=(tf.shape(observations)[0],tf.shape(observations)[1],
                                                          tf.shape(observations)[2],tf.shape(observations)[3]),
                                            padding=padding)
            next_observations.append(output)
        next_observations=tf.stack(next_observations,axis=1)
        # predict the next rewards using the reward model
        next_rewards=[]
        for i in range(next_states.shape[1]):
            output=self.reward_model.output(next_states[:,i])
            next_rewards.append(output)
        next_rewards=tf.stack(next_rewards,axis=1)
        # initialize the values and policies with zeros
        values=tf.zeros((tf.shape(next_states)[0],1))
        policies=tf.zeros((tf.shape(next_states)[0],self.action_size))
        # loop over the horizon steps
        for i in range(self.horizon):
            # predict the value and policy using the value and policy models
            value=self.value_model.output(next_states[:,i])
            policy=self.policy_model.output(next_states[:,i])
            # update the values and policies with exponential moving average
            values=(1-0.1)*values+0.1*value
            policies=(1-0.1)*policies+0.1*policy
        # return the next observations, rewards, values and policies
        return next_observations,next_rewards,values,policies