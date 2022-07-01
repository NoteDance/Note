# Note
documentation:https://github.com/7NoteDancing/Note-documentation


If you done your neural network,you can use kernel to train.
simple example:
'''
import Note.create.kernel as k   #import kernel
import tensorflow as tf              #import core
import cnn as c                          #import your class's python file
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create class object
kernel=k.kernel(cnn)                 #start kernel
kernel.core=tf                           #use core
kernel.data(train_data,train_labels)   #input you data,if you have test data can transfer to kernel API data()
                                                          #data can be a list,[data1,data2,...,datan]
kernel.train(32,5)         #train neural network
                                                #batch: batch size
                                                #epoch:epoch
'''                                              


# Parallel optimization:
Use multithreading to optimize.
Note have two types of parallel optimization:
1. not parallel computing gradient and optimizing.
2. parallel computing gradient and optimizing.


# Multithreading：
simple example:
'''
import Note.create.kernel as k   #import kernel
import tensorflow as tf              #import core
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
kernel=k.kernel(your neural network object)   #start kernel
kernel.core=tf                            #use core
kernel.thread=7                        #thread count
kernel.data(train_data,train_labels)   #input you data
kernel.PO=1 or kernel.PO=2
kernel.thread_lock=threading.Lock()
class thread(threading.Thread):
	def run(self):
		kernel.train(32,5)
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(thread count):
	_thread.join()
'''

Support stop and save in multithreading training.
simple example:
'''
import Note.create.kernel as k   #import kernel
import tensorflow as tf              #import core
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
kernel=k.kernel(your neural network object)   #start kernel
kernel.core=tf                            #use core
kernel.stop=True
kernel.file_path='example'
kernel.end_loss=0.7
kernel.end_flag=True
kernel.thread=7                        #thread count
kernel.data(train_data,train_labels)   #input you data
kernel.PO=1 or kernel.PO=2
kernel.thread_lock=threading.Lock()
class thread(threading.Thread):
	def run(self):
		kernel.train(32,5)
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(thread count):
	_thread.join()
'''
