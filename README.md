# Note


# documentation:
https://github.com/NoteDancing/Note-documentation/tree/main/Note%205.0%20documentation


# Introduction:
Note is a system for deep learning and reinforcement learning.It makes it easy to create and train neural network.It can speed up the training of neural network by multithreading.


# Deep Learning:
If you accomplish your neural network,you can use kernel to train.

neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%205.0%20documentation/DL/neural%20network

example:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train)   #input you data
kernel.train(32,5)         #train neural network
                           #batch: batch size
                           #epoch:epoch
kernel.save()              #save neural network
```
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
kernel=k.kernel()                 #start kernel
kernel.platform=tf                    #use platform
kernel.data(x_train,y_train)   #input you data
kernel.restore('save.dat')     #restore neural network
kernel.train(32,1)             #train again
```

pytorch:
```python
import Note.create.DL.kernel as k   #import kernel
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
                           #batch: batch size
                           #epoch:epoch
```


## Parallel optimization:
You can use parallel optimization to speed up neural network training,parallel optimization speed up training by multithreading.

Note have two types of parallel optimization:
1. not parallel computing gradient and optimizing.(kernel.PO=1)
2. parallel computing gradient and optimizing.(kernel.PO=2)

parallel optimization may cause unstable training(the estimate of the gradient is biased) but it can speed up training and make the loss function jump out of the local minimum.

neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%205.0%20documentation/DL/neural%20network

**Use second parallel optimization to train on MNIST,speed was increased by more than 2 times!**

**Tensorflow version:2.9.1**

**batch size:32**

**epoch:6**

**thread count:2**

**PO:2**

**CPU:i5-8400**

**Not use parallel optimization to train spending 15s,use parallel optimization to train spending 6.8s.**

![1](https://github.com/NoteDancing/Note-documentation/blob/main/picture/time.png)
![2](https://github.com/NoteDancing/Note-documentation/blob/main/picture/time(PO).png)

## Multithreading：
**Note can speed up training by multithreading and has stop mechanism to resolve unstable training.**

**Note uses multithreading parallel forward propagation and optimizes neural network.**

neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%205.0%20documentation/DL/neural%20network

example:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=2                        #thread count
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```

pytorch:
```python
import Note.create.DL.kernel as k   #import kernel
import torch                         #import platform
import nn as n                        #import neural network
import threading
from torchvision import datasets
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_data,train_labels=training_data
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=torch                            #use platform
kernel.thread=2                        #thread count
kernel.PO=1
kernel.data(train_data,train_labels)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```

Stop multithreading training and saving when condition is met.

example:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.stop=True
kernel.end_loss=0.7
kernel.thread=2                        #thread count
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```

pytorch:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
from torchvision import datasets
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_data,train_labels=training_data
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=torch                            #use platform
kernel.stop=True
kernel.end_loss=0.7
kernel.thread=2                        #thread count
kernel.PO=1
kernel.data(train_data,train_labels)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```


# Reinforcement Learning:
neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%205.0%20documentation/RL/neural%20network

example:
```python
import Note.create.RL.nspn.kernel as k   #import kernel
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn)   #start kernel
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.train(500)
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```
```python
import Note.create.RL.nspn.kernel as k   #import kernel
import DDPG as d
ddpg=d.DDPG(64,0.01,0.98,0.005,5e-4,5e-3)         #create neural network object
kernel=k.kernel(ddpg)   #start kernel
kernel.set_up(pool_size=10000,batch=64)
kernel.train(200)
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```


## Pool Network:
neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%205.0%20documentation/RL/neural%20network

example:
```python
import Note.create.RL.kernel as k   #import kernel
import DQN as d
import threading
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock(),threading.Lock()]
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(5):
	_thread=thread()
	_thread.start()
for _ in range(5):
	_thread.join()
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

Stop multithreading training and saving when condition is met.

example:
```python
import Note.create.RL.kernel as k   #import kernel
import DQN as d
import threading
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock(),threading.Lock(),threading.Lock()]
kernel.stop=True
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200)
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(5):
	_thread=thread()
	_thread.start()
for _ in range(5):
	_thread.join()
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```


# Note Compiler:
documentation:https://github.com/NoteDancing/Note-documentation/tree/main/Note%205.0%20documentation/compiler
```python
import Note.create.nc as nc
c=nc.compiler('nn.n')
c.Compile()
```


# Patreon:
You can support this project on Patreon.

https://www.patreon.com/NoteDancing
