# Note
## documentation:https://github.com/7NoteDancing/Note-documentation/blob/main/Note%204.0%20documentation/DL/kernel.txt


# Introduction:
Note is an AI system that have kernel for deep learning and reinforcement learning.It retains the freedom of tensorflow to implement neural networks,eliminates a lot of tedious work and has many functions.


# Train:
If you done your neural network,you can use kernel to train.

simple example:

neural network example:https://github.com/7NoteDancing/Note-documentation/tree/main/Note%204.0%20documentation/DL
```python
import Note.create.kernel as k   #import kernel
import tensorflow as tf              #import core
import cnn as c                          #import your class's python file
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create class object
kernel=k.kernel(cnn)                 #start kernel
kernel.core=tf                           #use core
kernel.data(x_train,y_train)   #input you data,if you have test data can transfer to kernel API data()
                                                          #data can be a list,[data1,data2,...,datan]
kernel.train(32,5)         #train neural network
                                                #batch: batch size
                                                #epoch:epoch
```                                             


# Parallel optimization:
You can use parallel optimization to speed up neural network training,parallel optimization speed up training by multithreading.

Note have two types of parallel optimization:
1. not parallel computing gradient and optimizing.(kernel.PO=1)
2. parallel computing gradient and optimizing.(kernel.PO=2)

Second parallel optimization may cause training instability but it can make the loss function jump out of the local minimum.

neural network example:https://github.com/7NoteDancing/Note-documentation/tree/main/Note%204.0%20documentation/DL

**Use second parallel optimization to train on MNIST,speed was increased by more than 2 times!**

**Tensorflow version:2.9.1**

**batch size:32**

**epoch:6**

**thread count:2**

**PO:2**

**GPU:GTX 1050 Ti**

**Not use parallel optimization to train spending 15s,use parallel optimization to train spending 6.8s.**

![1](https://github.com/7NoteDancing/Note-documentation/blob/main/1.png)
![2](https://github.com/7NoteDancing/Note-documentation/blob/main/2.png)

**Use second parallel optimization to train on CIFAR10,speed was increased by more than 1.2 times,loss was reduced by 34 percent.**

**Tensorflow version:2.9.1**

**batch size:32**

**epoch:10**

**thread count:5**

**PO:2**

**GPU:GTX 1050 Ti**


# Multithreading：
neural network example:https://github.com/7NoteDancing/Note-documentation/tree/main/Note%204.0%20documentation/DL

simple example:
```python
import Note.create.kernel as k   #import kernel
import tensorflow as tf              #import core
import cnn as c                          #import your class's python file
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create class object
kernel=k.kernel(cnn)   #start kernel
kernel.core=tf                            #use core
kernel.thread=2                        #thread count
kernel.data(x_train,y_train)   #input you data
kernel.PO=2
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
kernel.train_visual()
```

Support stop and save in multithreading training.

simple example:
```python
import Note.create.kernel as k   #import kernel
import tensorflow as tf              #import core
import cnn as c                          #import your class's python file
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create class object
kernel=k.kernel(cnn)   #start kernel
kernel.core=tf                            #use core
kernel.stop=True
kernel.save_flag=True
kernel.end_loss=0.7
kernel.end_flag=True
kernel.thread=2                        #thread count
kernel.data(x_train,y_train)   #input you data
kernel.PO=2
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
kernel.train_visual()
```
