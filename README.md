# Introduction:
- **Note is a system for deep learning and reinforcement learning, supporting TensorFlow and PyTorch platforms, supporting non-parallel training and parallel training. Note makes the building and training of neural networks easy. To train a neural network on Note, you first need to write a neural network class, pass the neural network object to the kernel, and then use the methods provided by the kernel to train a neural network. Note is based on the multiprocessing module of Python to implement parallel training. Because Note is based on the multiprocessing module of Python to implement parallel training, the number of parallel processes is related to the CPU. Note allows you to easily implement parallel training.**

- **The Note.nn.neuralnetwork package contains neural networks that can be trained on Note. You only need to provide the training data and operate simply, then you can train these neural networks in parallel on Note.**


# Installation:
**To use Note, you need to download it from https://github.com/NoteDancing/Note and then unzip it to the site-packages folder of your Python environment.**

**To import the neural network example conveniently, you can download it from https://github.com/NoteDancing/Note-documentation/tree/neural-network-example and unzip the neuralnetwork package to the site-packages folder of your Python environment.**


# Neural network:
**The neuralnetwork package in Note has models that can be trained in parallel on Note, such as ConvNeXt, EfficientNetV2, EfficientNet, etc. Currently, the performance of Note’s parallel training has not been sufficiently tested, if you want you can use the models in the neuralnetwork package to test Note’s parallel training performance.**

https://github.com/NoteDancing/Note/tree/Note-7.0/Note/nn/neuralnetwork

**Documentation:** https://github.com/NoteDancing/Note-documentation/tree/neural-network

**The following is an example of training on the CIFAR10 dataset.**

**ConvNeXtV2:**
```python
import Note.DL.parallel.kernel as k   #import kernel module
from Note.nn.neuralnetwork.ConvNeXtV2 import ConvNeXtV2 #import neural network class
from tensorflow.keras import datasets
from multiprocessing import Process,Manager #import multiprocessing tools
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
train_images,test_images=train_images/255.0,test_images/255.0
convnext_atto=ConvNeXtV2(model_type='atto',classes=10)  #create neural network object
convnext_atto.build()                           #build the network structure
kernel=k.kernel(convnext_atto)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(train_images,train_labels)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
```
**Use the trained model.**
```python
convnext_atto.km=0
output=convnext_atto.fp(data)
```
**EfficientNetV2B0:**
```python
import Note.DL.parallel.kernel as k   #import kernel module
from Note.nn.neuralnetwork.EfficientNetV2 import EfficientNetV2 #import neural network class
from tensorflow.keras import datasets
from multiprocessing import Process,Manager #import multiprocessing tools
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
efficientnetv2b0=EfficientNetV2(input_shape=[32,32,32,3],model_name='efficientnetv2-b0',classes=10)  #create neural network object
efficientnetv2b0.build()                           #build the network structure
kernel=k.kernel(efficientnetv2b0)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(train_images,train_labels)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
```
**Use the trained model.**
```python
efficientnetv2b0.km=0
output=efficientnetv2b0.fp(data)
```


# Create neural network:
- **Every neural network is regarded as an object, and the neural network object is passed into the kernel and trained by the kernel. To build a neural network that can be trained on Note, you need to follow some rules, otherwise you will get errors during training. You can see the examples of neural networks in the documentation. You can first learn the rules from the simple neural network examples named nn.py, nn_acc.py, and nn_device.py. Then, you can write a Python module for your neural network class and import it. Next, pass the neural network object to the kernel and train it.**

- **Neural network class should define a forward propagation function fp(data), and if using parallel kernel, it should define fp(data,p) where p is the process number. fp passes data and returns output, a loss function loss(output,labels), and if using parallel kernel, it should define loss(output,labels,p) where p is the process number. loss passes output and labels and returns loss value. If using parallel kernel, it should also define an optimization function opt(gradient,p) and GradientTape(data,labels,p) where p is the process number. opt passes gradient and returns parameter, GradientTape passes data and labels and returns tape, output and loss.**

**Examples of training neural networks with kernel are shown below.**


# Deep Learning:

## Non-parallel training:

### Tensorflow platform:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
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
```python
import Note.DL.kernel as k   #import kernel module
import torch                 #import torch library
import neuralnetwork.DL.pytorch.non_parallel.nn as n   #import neural network module
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

**2. Perform forward propagation, one gradient calculation or multiple gradient computations and optimization in parallel. (PO2)**

**3. Perform forward propagation, gradient calculation and optimization in parallel without locks. (PO3)**

- **Parallel optimization may cause unstable training(the estimate of the gradient is biased) but it can speed up training and make the loss function jump out of the local minimum. Note can speed up training by multiprocessing and has stop mechanism, gradient attenuation, and process priority to resolve unstable training. Note uses multiprocessing to perform parallel forward propagation and optimization on neural networks.**
- **Neural networks built with Keras may not be able to train in parallel on Note’s parallel kernel. You can use the layer modules in the Note.nn.layer package and the low-level API of Tensorflow to build neural networks that can train in parallel on Note’s parallel kernel. Do not use Keras’s optimizers because they cannot be serialized by the multiprocessing module. You can use the optimizers in the Note.nn.parallel package, or implement your own optimizers with the low-level API of Tensorflow.**

### Tensorflow platform:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

**Multidevice:**

**If you have multiple devices that you want to allocate, you can use the process index to freely assign devices to your operations.**
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn_device as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### Pytorch platform:
```python
import Note.DL.parallel.kernel_pytorch as k   #import kernel module
from torchvision import datasets
from torchvision.transforms import ToTensor
import neuralnetwork.DL.pytorch.parallel.nn as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
nn=n.neuralnetwork()                            #create neural network object
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=64                      #set the batch size
kernel.data(training_data)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
    Process(target=kernel.train,args=(p,)).start()
```


# Reinforcement Learning:

**The version of gym used in the example is less than 0.26.0.**

## Non-parallel training:

### Tensorflow platform:
```python
import Note.RL.kernel as k   #import kernel module
import tensorflow as tf           #import tensorflow library
import neuralnetwork.RL.tensorflow.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)                #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)              #create kernel object with the network
kernel.platform=tf                #set the platform to tensorflow
kernel.action_count=2             #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.train(100)                 #train the network for 100 episodes
kernel.visualize_train()
kernel.visualize_reward()
```
```python
import Note.RL.kernel as k   #import kernel module
import tensorflow as tf           #import tensorflow library
import neuralnetwork.RL.tensorflow.non_parallrl.DDPG as d   #import deep deterministic policy gradient module
ddpg=d.DDPG(64,0.01,0.98,0.005,5e-4,5e-3) #create neural network object with 64 inputs, 0.01 learning rate, 0.98 discount factor, 0.005 noise scale, 5e-4 actor learning rate and 5e-3 critic learning rate
kernel=k.kernel(ddpg)             #create kernel object with the network
kernel.platform=tf                #set the platform to tensorflow
kernel.set_up(pool_size=10000,batch=64) #set up the hyperparameters for training
kernel.train(200)                 #train the network for 200 episodes
kernel.visualize_train()
kernel.visualize_reward()
```

### Pytorch platform:
```python
import Note.RL.kernel as k   #import kernel module
import torch                      #import torch library
import neuralnetwork.RL.pytorch.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)                #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)              #create kernel object with the network
kernel.platform=torch             #set the platform to torch
kernel.action_count=2             #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.train(100)                 #train the network for 100 episodes
kernel.visualize_train()
kernel.visualize_reward()
```


## Parallel training:

**Pool Network:**

![3](https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/picture/Pool%20Network.png)

**Pool net use multiprocessing parallel and random add episode in pool,which would make data being uncorrelated in pool,
then pools would be used parallel training agent.**

### Tensorflow platform:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock()]         #create two locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```

### Pytorch platform:
```python
import Note.RL.parallel.kernel_pytorch as k   #import kernel module
import neuralnetwork.RL.pytorch.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock()]         #create two locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start()
```


# Study kernel:
**If you want to study kernel, you can see the kernel with comments at the link below.**

**DL:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/DL/kernel

**RL:** https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/RL/kernel


# Documentation:
**The document has kernel code and other code with comments that can help you understand.**

**Documentation readme has other examples.**

https://github.com/NoteDancing/Note-documentation


# Layer modules:
https://github.com/NoteDancing/Note/tree/Note-7.0/Note/nn/layer

**Documentation:** https://github.com/NoteDancing/Note-documentation/tree/layer


# Create a layer module for use on Note:
**To create a layer module for use on Note, you need to follow these steps:**

**1. Import the necessary modules from Note and TensorFlow.**

**2. Define a layer class.**

**3. In the init method, initialize the variables and submodules that you need for your layer. You can use the Note.nn.initializer module to create different types of initializers for your variables. You can also use the Note.nn.activation module to get different types of activation functions for your layer.**

**4. In the init method, create a list called self.param and append the variables and submodules of your layer to it. This will allow you to access and update the parameters later, such as during training or saving. It will also help you keep track of the trainable and non-trainable variables of your layer.**

**5. In the output method, define the logic of your layer. You can use any TensorFlow operations on the input tensors and the variables or submodules of your layer. You can also use any custom functions that you define outside the class. You should return the output tensor of your layer from this method.**

**You can refer to the layer module implementation in this link.** https://github.com/NoteDancing/Note/tree/Note-7.0/Note/nn/layer

**You can commit the layer module you write to Note.**


# Create an optimizer for use on Note:
**To create an optimizer for use on Note, you need to follow these steps:**

**1. Import the necessary modules from Note and TensorFlow.**

**2. Define an optimizer class.**

**3. In the init method, initialize the hyperparameters and internal states that you need for your optimizer.**

**4. In the opt method, define the logic of your optimizer. You can use any TensorFlow operations on the gradients and parameters of your model. You can also use any custom functions that you define outside the class. You should update the parameters of your model in place and return them from this method.**

**To create a parallel optimizer in Note, you need to follow these additional steps:**

**1. Import the multiprocessing. Manager module from Python. This module provides a way to create and manage shared objects across different processes.**

**2. In the init method, create a manager object by calling Manager(). This object will allow you to create shared lists for your internal states.**

**3. In the init method, create shared lists for your internal states by calling manager.list(). These lists will be accessible and modifiable by multiple processes.**

**4. In the opt method, use the shared lists instead of regular lists for your internal states. You can use the same operations on the shared lists as on regular lists.**

**You can refer to the optimizer implementations in the following two links.** 

**non-parallel:** https://github.com/NoteDancing/Note/blob/Note-7.0/Note/nn/optimizer.py

**parallel:** https://github.com/NoteDancing/Note/blob/Note-7.0/Note/nn/parallel/optimizer.py

**You can commit the optimizer you write to Note.**


# Patreon:
**You can support this project on Patreon.**

https://www.patreon.com/NoteDancing
