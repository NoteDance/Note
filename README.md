# Introduction:
**Note is a system(library) for deep learning and reinforcement learning. Note makes the building and training of neural networks easy and flexible. Note.nn.layer package contains many layer modules, you can use them to build neural networks. Note’s layer modules are implemented based on TensorFlow, which means they are compatible with TensorFlow’s API. The layer modules allow you to build neural networks in the style of PyTorch or Keras. You can not only use the layer modules to build neural networks trained on Note but also use them to build neural networks trained with TensorFlow.**


# Installation:
**To use Note, you need to download it from https://github.com/NoteDance/Note and then unzip it to the site-packages folder of your Python environment.**


# Layer modules:
**To use the layer module, you first need to create a layer object, then input data to get the output, like using pytorch, or you can use the layer module like using keras. The args of the layer classes in Note are similar to those of the layer classes in tf.keras.layers, so you can refer to the API documentation of tf.keras.layers to use the layer classes in Note. Neural networks created with the layer module are compatible with TensorFlow, which means you can train these neural networks with TensorFlow.**

https://github.com/NoteDance/Note/tree/Note-7.0/Note/nn/layer

**Documentation:** https://github.com/NoteDance/Note-documentation/tree/layer-7.0

**Using Note’s Layer module, you can determine the shape of the training parameters when you input data like Keras, or you can give the shape of the training parameters in advance like PyTorch.**

**Pytorch:**
```python
from Note import nn

class nn(nn.Model):
    def __init__(self):
	super().__init__()
        self.layer1=nn.dense(128,784,activation='relu')
        self.layer2=nn.dense(10,128)
    
    def __call__(self,data):
        x=self.layer1(data)
        x=self.layer2(x)
        return x
```
**Keras:**
```python
from Note import nn

class nn(nn.Model):
    def __init__(self):
	super().__init__()
        self.layer1=nn.dense(128,activation='relu')
        self.layer2=nn.dense(10)
    
    def __call__(self,data):
        x=self.layer1(data)
        x=self.layer2(x)
        return x
```
```python
from Note import nn

nn.Model.init()
def nn(data):
    x=nn.dense(128,activation='relu')(data)
    x=nn.dense(10)(x)
    return x
```
**Note.neuralnetwork.tf package contains neural networks implemented with Note’s layer module that can be trained with TensorFlow.**
https://github.com/NoteDance/Note/tree/Note-7.0/Note/neuralnetwork/tf

**Documentation:** https://github.com/NoteDance/Note-documentation/tree/tf-7.0


# Note.nn.Model.Model:
**Model class manages the parameters of the neural network.**

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/Model.py


# Note.nn.initializer.initializer:
**This function is used to initialize the parameters of the neural network, and it returns a TensorFlow variable.**

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/initializer.py


# Note.nn.initializer.initializer_:
**This function is used to initialize the parameters of the neural network, it returns a TensorFlow variable and stores the variable in Module.param.**

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/initializer.py


# Note.nn.Layers.Layers:
**This class is used similarly to the tf.keras.Sequential class.**

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/Layers.py


# The models that can be trained with TensorFlow:
**This package include Llama2, CLIP, ViT, ConvNeXt, SwiftFormer, etc. These models built with Note are compatible with TensorFlow and can be trained with TensorFlow.**

https://github.com/NoteDance/Note/tree/Note-7.0/Note/neuralnetwork/tf

**Documentation:** https://github.com/NoteDance/Note-documentation/tree/tf-7.0


# Documentation:
**The document contains tutorials on how to build neural networks that can be trained on Note, how to train neural networks with Note, and kernel code as well as other code with comments that can help you understand.**

https://github.com/NoteDance/Note-documentation


# Patreon:
**You can support this project on Patreon.**

https://www.patreon.com/NoteDance


# Contact:
**If you have any issues with the use, or you have any suggestions, you can contact me.**

**E-mail:** notedance@outlook.com
