# Note


# Introduction:
Note is a system for deep learning and reinforcement learning.It makes it easy to create and train neural network.Note supports TensorFlow and PyTorch platform.It can speed up the training of neural network by multithreading and multiprocessing.


# Installation:
To use Note you need to download it and then unzip it to site-packages folder.


# Create neural network:
You need to create your neural network according to some rules, otherwise you may get AttributeError or other exceptions when you train with kernel.You can refer to the neural network examples in the documentation to create your neural network.

neural network example:

You can refer to cnn, cnn_acc, nn, nn_ in the documentation.

**DL:** https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv%20documentation/DL/neural%20network

**RL:** https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv%20documentation/RL/neural%20network

If you accomplish your neural network,you can use kernel to train,examples are shown below.


# Deep Learning:

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/cnn.py

## Tensorflow platform:

**example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)                 #start kernel
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
import cnn_acc as c                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train,x_test,y_test)   #input you data
kernel.train(32,5,32)         #train neural network
                           #batch size:32
			   #test batch size:32
                           #epoch:5
kernel.save()              #save neural network
```

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/lstm.py

**example(lstm):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import tensorflow_datasets as tfds
import lstm as l                          #import neural network
dataset,info=tfds.load('imdb_reviews',with_info=True,as_supervised=True)
train_dataset=dataset['train']
train_dataset=train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
VOCAB_SIZE=1000
encoder=tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text,label:text))
lstm=l.lstm(encoder)                                #create neural network object
kernel=k.kernel(lstm)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(train_dataset=train_dataset)   #input you data
kernel.train(64,10)         #train neural network
                           #batch size:64
                           #epoch:10
kernel.save()              #save neural network
```

### Parallel optimization:
**You can use parallel optimization to speed up neural network training,parallel optimization speed up training by multiprocessing or multithreading.**

**Note have three types of parallel optimization:**

**1. parallel forward propagation and optimizing.(kernel.PO=1)**

**2. parallel forward propagation and calculate a gradient and optimizing.(kernel.PO=2)**

**3. parallel forward propagation and calculate multiple gradients and optimizing.(kernel.PO=3)**

**parallel optimization may cause unstable training(the estimate of the gradient is biased) but it can speed up training and make the loss function jump out of the local minimum.**

**Use second parallel optimization to train on MNIST,speed was increased by more than 2 times!Not use parallel optimization to train spending 15s,use parallel optimization to train spending 6.8s.**

**Tensorflow version:2.9.1**

**batch size:32**

**epoch:6**

**thread count:2**

**PO:2**

**CPU:i5-8400**

#### Multithreading:
**Note can speed up training by multithreading and has stop mechanism and gradient attenuation to resolve unstable training.**

**Note uses multithreading parallel forward propagation and optimizes neural network.**

**multithreading example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=7                        #thread count,use 7 threads to train
kernel.epoch_=6                #epoch:6
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32) #batch size:32
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
kernel.visualize_train()
```

**multithreading example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=2                        #thread count,use 2 threads to train
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):     
	def run(self):              
		kernel.train(32,3)  #batch size:32 epoch:3
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
kernel.save()              #save neural network
```
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
kernel=k.kernel()                 #start kernel
kernel.platform=tf                    #use platform
kernel.restore('save.dat')     #restore neural network
kernel.process_thread=2                        #thread count,use 2 threads to train
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,1) #batch size:32 epoch:1
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
```

#### Multiprocessing:
**multiprocessing example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
from multiprocessing import Process,Lock
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=7                        #thread count,use 7 processes to train
kernel.epoch_=6                #epoch:6
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[Lock(),Lock(),Lock()]
for _ in range(7):
	p=Process(target=kernel.train(32)) #batch size:32
	p.start()
for _ in range(7):
	p.join()
kernel.visualize_train()
```
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
from multiprocessing import Process,Lock
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=7                        #thread count,use 7 processes to train
kernel.data_segment_flag=True
kernel.batches=1875
kernel.epoch_=6                #epoch:6
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[Lock(),Lock(),Lock()]
for _ in range(7):
	p=Process(target=kernel.train(32)) #batch size:32
	p.start()
for _ in range(7):
	p.join()
kernel.visualize_train()
```

#### Use threads in process：
**example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
from multiprocessing import Process
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=[3,7]                        #use 3 processes and 21 threads to train
kernel.epoch_=6                #epoch:6
kernel.PO=2
kernel.multiprocessing_threading=threading
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
for _ in range(3):
	p=Process(target=kernel.train(32)) #batch size:32
	p.start()
for _ in range(3):
	p.join()
```

**Gradient Attenuation：**

**Calculate the attenuation coefficient based on the optimization counter using the attenuation function.**

**example:https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/cnn_a.py**

**picture:https://github.com/NoteDancing/Note-documentation/tree/main/picture/gradient%20attenuation**

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/DL/neural%20network/pytorch/nn.py

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

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/RL/neural%20network/pytorch/DQN.py

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
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/DQN.py

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
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/DDPG.py

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
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

## Pool Net:
![3](https://github.com/NoteDancing/Note-documentation/blob/main/picture/Pool%20Net.png)

**Pool net use multithreading parallel and random add episode in pool,which would make data being uncorrelated in pool,
then pools would be used parallel training agent.**

**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/pool%20net/DQN.py

**multithreading example:**
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
import threading
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=1
kernel.multiprocessing_threading=threading
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
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

**multithreading example:**
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
import threading
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=3
kernel.multiprocessing_threading=threading
kernel.max_lock=5
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
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


# Parallel test:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/cnn_acc.py
```python
import cnn_acc as c
import Note.DL.dl.test as t
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
test=t.test_pt(cnn,x_test,y_test,6,32)
class thread(threading.Thread):     
	def run(self):              
		test.test()
for _ in range(6):
	_thread=thread()
	_thread.start()
for _ in range(6):
	_thread.join()
loss,acc=test.loss_acc()
```


# Note Compiler:
documentation:https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv%20documentation/compiler
```python
import Note.nc as nc
c=nc.compiler('nn.n')
c.Compile()
```


# documentation:
https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv%20documentation


# Patreon:
You can support this project on Patreon.

https://www.patreon.com/NoteDancing
