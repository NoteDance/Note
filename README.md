# Note


# Introduction:
Note is a system for deep learning and reinforcement learning.It makes it easy to create and train neural network.Note supports TensorFlow and PyTorch platform.It can speed up neural network training by multiprocessing.


# Installation:
To use Note, you need to download it from https://github.com/NoteDancing/Note and then unzip it to the site-packages folder of your Python environment.


# Create neural network:
You need to create your neural network according to some rules, otherwise you may get AttributeError or other exceptions when you train with kernel.You can refer to the neural network examples in the documentation to create your neural network. You can write your neural network class as a python module and import it, or you can write a neural network class directly in the python interpreter. When you import your neural network or you write a neural network class, you can pass the neural network object to the kernel and use the kernel to train the neural network.

neural network example:

You can first refer to the simpler neural network examples nn.py and nn_acc.py in the documentation.

## DL: 
https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/DL/neural%20network

**Neural network examples using Note's layer module:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/layer

**Neural network examples for Note multi-process kernel:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/process

## RL: 
https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/RL/neural%20network

If you accomplish your neural network,you can use kernel to train,examples are shown below.


# Deep Learning:

## Non-parallel training:

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/nn.py

### Tensorflow platform:

**example:**
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import nn as n               #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train) #input train data to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
kernel.test(x_test,y_test,32)#test the network performance on the test set with batch size 32
```

### Pytorch platform:

**example:**
```python
import Note.DL.kernel as k   #import kernel module
import torch                 #import torch library
import nn as n               #import neural network module
from torch.utils.data import DataLoader #import data loader tool
from torchvision import datasets        #import datasets tool
from torchvision.transforms import ToTensor #import tensor transformation tool
training_data=datasets.FashionMNIST(    #load FashionMNIST dataset
    root="data",                        #set the root directory for data
    train=True,                         #set the flag to use train data
    download=True,                      #set the flag to download data if not available
    transform=ToTensor(),               #set the transformation to convert images to tensors
)
train_dataloader=DataLoader(training_data,batch_size=60000) #create a data loader object with the training data and batch size 60000
for train_data,train_labels in train_dataloader: #loop over the data loader
    break                                      #break the loop after getting one batch of data and labels
nn=n.neuralnetwork()                          #create neural network object
kernel=k.kernel(nn)                           #create kernel object with the network
kernel.platform=torch                         #set the platform to torch
kernel.data(train_data,train_labels)          #input train data and labels to the kernel
kernel.train(64,5)                            #train the network with batch size 64 and epoch 5
```


## Parallel training:

**Parallel optimization:**

**You can use parallel optimization to speed up neural network training, and parallel optimization is implemented through multiprocessing.**

**Note have three types of parallel optimization:**

**1. Perform forward propagation and optimization in parallel. (PO1)**

**2. Perform forward propagation, one gradient computation or multiple gradient computations and optimization in parallel. (PO2)**

**3. Perform forward propagation, gradient computation and optimization in parallel without locks. (PO3)**

**Parallel optimization may cause unstable training(the estimate of the gradient is biased) but it can speed up training and make the loss function jump out of the local minimum. Note can speed up training by multiprocessing and has stop mechanism and gradient attenuation to resolve unstable training. Note uses multiprocessing to perform parallel forward propagation and optimization on neural networks. Note's multi-process kernel is not compatible with the neural network built by Keras. You can use the layer directory from Note and the low-level API from tensorflow to build neural networks.**

### Tensorflow platform:

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/process/nn.py

**example:**
```python
import Note.DL.process.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=1                          #use PO1 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```
**example(process priority):**
```python
import Note.DL.process.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.priority_flag=True            #set the flag to use priority scheduling for processes
kernel.PO=1                          #use PO1 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

**Gradient Attenuation：**

**Calculate the attenuation coefficient based on the optimization counter using the attenuation function.**

**example:https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/process/nn_attenuate.py**

### Pytorch platform:

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/pytorch/process/nn.py

**example:**
```python
import Note.DL.process.kernel_pytorch as k   #import kernel module
from torchvision import datasets
from torchvision.transforms import ToTensor
import nn as n                       #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
nn=n.neuralnetwork()                            #create neural network object
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=64                      #set the batch size
kernel.data(training_data)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(7):                   #loop over the processes
    Process(target=kernel.train,args=(p,)).start()
```


# Reinforcement Learning:

**The version of gym used in the example is less than 0.26.0.**

## Non-parallel training:

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/RL/neural%20network/tensorflow/DQN.py

### Tensorflow platform:

**example:**
```python
import Note.RL.nspn.kernel as k   #import kernel module
import tensorflow as tf           #import tensorflow library
import DQN as d                   #import deep Q-network module
dqn=d.DQN(4,128,2)                #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)              #create kernel object with the network
kernel.platform=tf                #set the platform to tensorflow
kernel.action_count=2             #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.train(500)                 #train the network for 500 episodes
kernel.visualize_train()
kernel.visualize_reward()

```

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/RL/neural%20network/tensorflow/DDPG.py

**example:**
```python
import Note.RL.nspn.kernel as k   #import kernel module
import tensorflow as tf           #import tensorflow library
import DDPG as d                  #import deep deterministic policy gradient module
ddpg=d.DDPG(64,0.01,0.98,0.005,5e-4,5e-3) #create neural network object with 64 inputs, 0.01 learning rate, 0.98 discount factor, 0.005 noise scale, 5e-4 actor learning rate and 5e-3 critic learning rate
kernel=k.kernel(ddpg)             #create kernel object with the network
kernel.platform=tf                #set the platform to tensorflow
kernel.set_up(pool_size=10000,batch=64) #set up the hyperparameters for training
kernel.train(200)                 #train the network for 200 episodes
kernel.visualize_train()
kernel.visualize_reward()
```

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/RL/neural%20network/pytorch/DQN.py

### Pytorch platform:

**example:**
```python
import Note.RL.nspn.kernel as k   #import kernel module
import torch                      #import torch library
import DQN as d                   #import deep Q-network module
dqn=d.DQN(4,128,2)                #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)              #create kernel object with the network
kernel.platform=torch             #set the platform to torch
kernel.action_count=2             #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.train(500)                 #train the network for 500 episodes
kernel.visualize_train()
kernel.visualize_reward()
```


## Parallel training:

**Pool Network:**

![3](https://github.com/NoteDancing/Note-documentation/blob/main/picture/Pool%20Net.png)

**Pool net use multiprocessing or multithreading parallel and random add episode in pool,which would make data being uncorrelated in pool,
then pools would be used parallel training agent.**

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/RL/neural%20network/tensorflow/pool%20net/DQN.py

**example:**
```python
import Note.RL.kernel as k   #import kernel module
import DQN as d              #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=1                  #use PO1 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create a list of locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```
**example(process priority):**
```python
import Note.RL.kernel as k   #import kernel module
import DQN as d              #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
manager=Manager()            #create manager object to share data among processes
kernel.priority_flag=True    #set the flag to use priority scheduling for processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=1                  #use PO1 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create a list of locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```


# Study kernel:
**If you want to study kernel, you can see the kernel with comments at the link below.**

**DL:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/DL/kernel

**RL:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/RL/kernel


# Documentation:
**Documentation's readme has other examples.**

https://github.com/NoteDancing/Note-documentation


# Layer:
https://github.com/NoteDancing/Note/tree/Note-7.0/Note/nn/layer

**documentation:** https://github.com/NoteDancing/Note-documentation/tree/layer

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/layer/nn.py
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import nn as n               #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
nn.build()                   #build the network structure
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train) #input train data and labels to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
```


# GPT:
**Layers in the GPT directory created by GPT**

https://github.com/NoteDancing/Note/tree/Note-7.0/Note/nn/layer/GPT


# Patreon:
You can support this project on Patreon.

https://www.patreon.com/NoteDancing
