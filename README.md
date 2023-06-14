# Note


# Introduction:
Note is a system for deep learning and reinforcement learning.It makes it easy to create and train neural network.Note supports TensorFlow and PyTorch platform.It can speed up neural network training by multiprocessing.


# Installation:
To use Note, you need to download it from https://github.com/NoteDancing/Note and then unzip it to the site-packages folder of your Python environment.


# Create neural network:
You need to create your neural network according to some rules, otherwise you may get AttributeError or other exceptions when you train with kernel.You can refer to the neural network examples in the documentation to create your neural network.

neural network example:

You can first refer to the simpler neural network examples nn.py and nn_acc.py in the documentation.

## DL: 
https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network

**Neural network examples using Note's layer module:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/layer

**Neural network examples for Note multi-process kernel:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/process

## RL: 
https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/Note%207.0%20pv%20documentation/RL/neural%20network

If you accomplish your neural network,you can use kernel to train,examples are shown below.


# Deep Learning:

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn.py

## Tensorflow platform:

**example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train)   #input you data
kernel.train(32,5)         #train neural network
                           #batch size:32
                           #epoch:5
kernel.save()              #save neural network
```
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
kernel=k.kernel()                 #start kernel
kernel.platform=tf                    #use platform
kernel.data(x_train,y_train)   #input you data
kernel.restore('save.dat')     #restore neural network
kernel.train(32,1)             #train again
```

**example(test):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn_acc as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train,x_test,y_test)   #input you data
kernel.train(32,5,32)         #train neural network
                           #batch size:32
			   #test batch size:32
                           #epoch:5
kernel.save()              #save neural network
```


### Parallel optimization:
**You can use parallel optimization to speed up neural network training,parallel optimization speed up training by multiprocessing.**

**Note have three types of parallel optimization:**

**1. parallel forward propagation and optimizing.(kernel.PO=1)**

**2. parallel forward propagation and calculate a gradient and optimizing.(kernel.PO=2)**

**3. parallel forward propagation and calculate multiple gradients and optimizing.(kernel.PO=3)**

**parallel optimization may cause unstable training(the estimate of the gradient is biased) but it can speed up training and make the loss function jump out of the local minimum.**

**Note can speed up training by multiprocessing and has stop mechanism and gradient attenuation to resolve unstable training.**

**Note uses multiprocessing parallel forward propagation and optimizes neural network.**

**Note's multi-process kernel is not compatible with the neural network built by Keras. You can use the layer package from Note and the low-level API from tensorflow to build neural networks.**

#### Multiprocessing:

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/process/nn.py

**multiprocessing example:**
```python
import Note.DL.process.kernel as k   #import kernel
import tensorflow as tf
import nn as n                          #import neural network
from multiprocessing import Process,Lock,Manager
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
nn.build()
kernel=k.kernel(nn)   #start kernel
kernel.process=7      #7 processes to train
kernel.data_segment_flag=True
kernel.epoch=6                #epoch:6
kernel.batch=32            #batch:32
kernel.PO=1                    #use PO1
kernel.data(x_train,y_train)   #input you data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=[Lock(),Lock(),Lock()]
for p in range(7):
	Process(target=kernel.train,args=(p,lock)).start()
kernel.update_nn_param()
kernel.test(x_train,y_train,32)
```
**multiprocessing example(process priority):**
```python
import Note.DL.process.kernel as k   #import kernel
import tensorflow as tf
import nn as n                          #import neural network
from multiprocessing import Process,Lock,Manager
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
nn.build()
kernel=k.kernel(nn)   #start kernel
kernel.process=7      #7 processes to train
kernel.data_segment_flag=True
kernel.epoch=6                #epoch:6
kernel.batch=32            #batch:32
kernel.priority_flag=True
kernel.PO=1                    #use PO1
kernel.data(x_train,y_train)   #input you data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=[Lock(),Lock(),Lock()]
for p in range(7):
	Process(target=kernel.train,args=(p,lock)).start()
kernel.update_nn_param()
kernel.test(x_train,y_train,32)
```

**Gradient Attenuation：**

**Calculate the attenuation coefficient based on the optimization counter using the attenuation function.**

**example:https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/process/nn_attenuate.py**

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/pytorch/nn.py

## Pytorch platform:

**example:**
```python
import Note.DL.kernel as k   #import kernel
import torch                         #import platform
import nn as n                          #import neural network
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_dataloader=DataLoader(training_data,batch_size=60000)
for train_data,train_labels in train_dataloader:
    break
nn=n.neuralnetwork()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=torch                   #use platform
kernel.data(train_data,train_labels)   #input you data
kernel.train(64,5)         #train neural network
                           #batch size:32
                           #epoch:5
```


# Reinforcement Learning:

**The version of gym used in the example is less than 0.26.0.**

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/RL/neural%20network/pytorch/DQN.py

## Pytorch platform:

**example:**
```python
import Note.RL.nspn.kernel as k   #import kernel
import torch
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn)   #start kernel
kernel.platform=torch
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.train(500)
kernel.visualize_train()
kernel.visualize_reward()
```

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/DQN.py

## Tensorflow platform:

**example:**
```python
import Note.RL.nspn.kernel as k   #import kernel
import tensorflow as tf
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn)   #start kernel
kernel.platform=tf
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.train(500)
kernel.visualize_train()
kernel.visualize_reward()
```

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/DDPG.py

**example:**
```python
import Note.RL.nspn.kernel as k   #import kernel
import tensorflow as tf
import DDPG as d
ddpg=d.DDPG(64,0.01,0.98,0.005,5e-4,5e-3)         #create neural network object
kernel=k.kernel(ddpg)   #start kernel
kernel.platform=tf
kernel.set_up(pool_size=10000,batch=64)
kernel.train(200)
kernel.visualize_train()
kernel.visualize_reward()
```

## Pool Network:
![3](https://github.com/NoteDancing/Note-documentation/blob/main/picture/Pool%20Net.png)

**Pool net use multiprocessing or multithreading parallel and random add episode in pool,which would make data being uncorrelated in pool,
then pools would be used parallel training agent.**

### Multiprocessing:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/pool%20net/DQN.py

**multiprocessing example:**
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
from multiprocessing import Process,Lock,Manager
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
manager=Manager()        #create manager object
kernel.init(manager)     #initialize shared data
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=1                    #use PO1
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()]
lock=[Lock(),Lock(),Lock()]
for p in range(5):
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start()
```
**multiprocessing example(process priority):**
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
from multiprocessing import Process,Lock,Manager
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
manager=Manager()        #create manager object
kernel.priority_flag=True
kernel.init(manager)     #initialize shared data
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=1                    #use PO1
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()]
lock=[Lock(),Lock(),Lock()]
for p in range(5):
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start()
```


# Study kernel:
**If you want to study kernel,you can start with reduced kernel,there is reduced kernel at link.**

**DL:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/reduced%20kernel/DL

**RL:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/reduced%20kernel/RL


# Documentation:
**Documentation's readme has other examples.**

https://github.com/NoteDancing/Note-documentation


# Layer:
https://github.com/NoteDancing/Note/tree/Note-7.0-pv/Note/nn/layer

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/layer/nn.py
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
nn.build()                          #build neural network
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train)   #input you data
kernel.train(32,5)         #train neural network
```


# GPT:
**Layers in the GPT directory created by GPT**

https://github.com/NoteDancing/Note/tree/Note-7.0-pv/Note/nn/layer/GPT


# Patreon:
You can support this project on Patreon.

https://www.patreon.com/NoteDancing
