import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class conv2d_bn:
    def __init__(self,input_channels,filters,kernel_size,activation="relu",use_bias=False,dtype='float32'):
        if type(kernel_size)==list:
            self.weight=initializer([kernel_size[0],kernel_size[1],input_channels,filters],'Xavier',dtype)
        else:
            self.weight=initializer([kernel_size,kernel_size,input_channels,filters],'Xavier',dtype)
        if activation!=None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        self.use_bias=use_bias
        self.param=[self.weight]
        if not use_bias:
            self.moving_mean=tf.zeros([filters])
            self.moving_var=tf.ones([filters])
            self.beta=tf.Variable(tf.zeros([filters]))
            self.param.append(self.beta)
        self.output_size=filters
    
    
    def output(self,data,strides=1,padding="SAME",train_flag=True):
        data=tf.nn.conv2d(data,self.weight,strides=strides,padding=padding)
        if train_flag:
            if not self.use_bias:
                mean,var=tf.nn.moments(data,axes=3,keepdims=True)
                self.moving_mean=self.moving_mean*0.99+mean*(1-0.99)
                self.moving_var=self.moving_var*0.99+var*(1-0.99)
                data=tf.nn.batch_normalization(data,mean,var,self.beta,None,1e-3)
        else:
            if not self.use_bias:
                data=tf.nn.batch_normalization(data,self.moving_mean,self.moving_var,self.beta,None,1e-3)
        if self.activation is not None:
            output=self.activation(data)
        else:
            output=data
        return output


class inception_resnet_block:
    def __init__(self,input_channels,scale,block_type,activation="relu",dtype='float32'):
        if block_type == "block35":
            self.branch_0 = conv2d_bn(input_channels, 32, 1, dtype=dtype)
            self.branch_1 = conv2d_bn(input_channels, 32, 1, dtype=dtype)
            self.branch_1_1 = conv2d_bn(self.branch_1.output_size, 32, 3, dtype=dtype)
            self.branch_2 = conv2d_bn(input_channels, 32, 1, dtype=dtype)
            self.branch_2_1 = conv2d_bn(self.branch_2.output_size, 48, 3, dtype=dtype)
            self.branch_2_2 = conv2d_bn(self.branch_2_1.output_size, 64, 3, dtype=dtype)
            output_size=self.branch_0.output_size+self.branch_1_1.output_size+self.branch_2_2.output_size
            self.param=[self.branch_0.param,self.branch_1.param,self.branch_1_1.param,self.branch_2.param,self.branch_2_1.param,self.branch_2_2.param]
        elif block_type == "block17":
            self.branch_0 = conv2d_bn(input_channels, 192, 1, dtype=dtype)
            self.branch_1 = conv2d_bn(input_channels, 128, 1, dtype=dtype)
            self.branch_1_1 = conv2d_bn(self.branch_1.output_size, 160, [1, 7], dtype=dtype)
            self.branch_1_2 = conv2d_bn(self.branch_1_1.output_size, 192, [7, 1], dtype=dtype)
            output_size=self.branch_0.output_size+self.branch_1_2.output_size
            self.param=[self.branch_0.param,self.branch_1.param,self.branch_1_1.param,self.branch_1_2.param]
        elif block_type == "block8":
            self.branch_0 = conv2d_bn(input_channels, 192, 1, dtype=dtype)
            self.branch_1 = conv2d_bn(input_channels, 192, 1, dtype=dtype)
            self.branch_1_1 = conv2d_bn(self.branch_1.output_size, 224, [1, 3], dtype=dtype)
            self.branch_1_2 = conv2d_bn(self.branch_1_1.output_size, 256, [3, 1], dtype=dtype)
            output_size=self.branch_0.output_size+self.branch_1_2.output_size
            self.param=[self.branch_0.param,self.branch_1.param,self.branch_1_1.param,self.branch_1_2.param]
        self.up=conv2d_bn(output_size,input_channels,1,activation=None,use_bias=True)
        self.param.append(self.up.param)
        self.scale=scale
        self.block_type=block_type
        if activation!=None:
            self.activation=activation_dict[activation]
        else:
            self.activation=None
        self.output_size=self.up.output_size
    

    def CustomScaleLayer(self,inputs,scale):
        return inputs[0] + inputs[1] * self.scale
    
    
    def output(self,data,train_flag=True):
        if self.block_type == "block35":
            data1 = self.branch_0.output(data,train_flag=train_flag)
            data2 = self.branch_1.output(data,train_flag=train_flag)
            data2 = self.branch_1_1.output(data2,train_flag=train_flag)
            data3 = self.branch_2.output(data,train_flag=train_flag)
            data3 = self.branch_2_1.output(data3,train_flag=train_flag)
            data3 = self.branch_2_2.output(data3,train_flag=train_flag)
            branches = [data1, data2, data3]
        elif self.block_type == "block17":
            data1 = self.branch_0.output(data,train_flag=train_flag)
            data2 = self.branch_1.output(data,train_flag=train_flag)
            data2 = self.branch_1_1.output(data2,train_flag=train_flag)
            data2 = self.branch_1_2.output(data2,train_flag=train_flag)
            branches = [data1, data2]
        elif self.block_type == "block8":
            data1 = self.branch_0.output(data,train_flag=train_flag)
            data2 = self.branch_1.output(data,train_flag=train_flag)
            data2 = self.branch_1_1.output(data2,train_flag=train_flag)
            data2 = self.branch_1_2.output(data2,train_flag=train_flag)
            branches = [data1, data2]
        mixed = tf.concat(branches,axis=3)
        up=self.up.output(mixed,train_flag=train_flag)
        x = self.CustomScaleLayer([data, up],self.scale)
        if self.activation is not None:
            output = self.activation(x)
        else:
            output=x
        return output


class InceptionResNetV2:
    def __init__(self,classes=1000,include_top=True,pooling=None,device='GPU'):
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.km=0
    
    
    def build(self,dtype='float32'):
        # Stem block: 35 x 35 x 192
        self.Stem_layer1 = conv2d_bn(3, 32, 3, dtype=dtype)
        self.Stem_layer2 = conv2d_bn(self.Stem_layer1.output_size, 32, 3, dtype=dtype)
        self.Stem_layer3 = conv2d_bn(self.Stem_layer2.output_size, 64, 3, dtype=dtype)
        self.Stem_layer4 = tf.nn.max_pool2d
        self.Stem_layer5 = conv2d_bn(self.Stem_layer3.output_size, 80, 1, dtype=dtype)
        self.Stem_layer6 = conv2d_bn(self.Stem_layer5.output_size, 192, 3, dtype=dtype)
        self.Stem_layer7 = tf.nn.max_pool2d
        self.param=[self.Stem_layer1.param,self.Stem_layer2.param,self.Stem_layer3.param,
                    self.Stem_layer5.param,self.Stem_layer6.param]
        
        # Mixed 5b (Inception-A block): 35 x 35 x 320
        self.branch_0_5b = conv2d_bn(self.Stem_layer6.output_size, 96, 1, dtype=dtype)
        self.branch_1_5b = conv2d_bn(self.Stem_layer6.output_size, 48, 1, dtype=dtype)
        self.branch_1_1_5b = conv2d_bn(self.branch_1_5b.output_size, 64, 5, dtype=dtype)
        self.branch_2_5b = conv2d_bn(self.Stem_layer6.output_size, 64, 1, dtype=dtype)
        self.branch_2_1_5b = conv2d_bn(self.branch_2_5b.output_size, 96, 3, dtype=dtype)
        self.branch_2_2_5b = conv2d_bn(self.branch_2_1_5b.output_size, 96, 3, dtype=dtype)
        self.branch_avg_5b = tf.nn.avg_pool2d
        self.branch_pool_5b = conv2d_bn(self.Stem_layer6.output_size, 64, 1, dtype=dtype)
        self.param.extend([self.branch_0_5b.param,self.branch_1_5b.param,self.branch_1_1_5b.param,
                           self.branch_2_5b.param,self.branch_2_1_5b.param,self.branch_2_2_5b.param,
                           self.branch_pool_5b.param])
        output_size=self.branch_0_5b.output_size+self.branch_1_1_5b.output_size+self.branch_2_2_5b.output_size+self.branch_pool_5b.output_size
        
        # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
        self.Inception_ResNet_A=[]
        for i in range(1, 11):
            if i==1:
                block = inception_resnet_block(output_size, scale=0.17, block_type="block35", dtype=dtype)
                self.Inception_ResNet_A.append(block)
                self.param.append(block.param)
            else:
                block = inception_resnet_block(self.Inception_ResNet_A[-1].output_size, scale=0.17, block_type="block35", dtype=dtype)
                self.Inception_ResNet_A.append(block)
                self.param.append(block.param)
        
        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        self.branch_0_6a = conv2d_bn(self.Inception_ResNet_A[-1].output_size, 384, 3, dtype=dtype)
        self.branch_1_6a = conv2d_bn(self.Inception_ResNet_A[-1].output_size, 256, 1, dtype=dtype)
        self.branch_1_1_6a = conv2d_bn(self.branch_1_6a.output_size, 256, 3, dtype=dtype)
        self.branch_1_2_6a = conv2d_bn(self.branch_1_1_6a.output_size, 384, 3, dtype=dtype)
        self.branch_pool_6a = tf.nn.max_pool2d
        self.param.extend([self.branch_0_6a.param,self.branch_1_6a.param,self.branch_1_1_6a.param,
                          self.branch_1_2_6a.param])
        output_size=self.branch_0_6a.output_size+self.branch_1_2_6a.output_size+self.Inception_ResNet_A[-1].output_size
        
        # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
        self.Inception_ResNet_B=[]
        for i in range(1, 21):
            if i==1:
                block = inception_resnet_block(output_size, scale=0.1, block_type="block17", dtype=dtype)
                self.Inception_ResNet_B.append(block)
                self.param.append(block.param)
            else:
                block = inception_resnet_block(self.Inception_ResNet_B[-1].output_size, scale=0.1, block_type="block17", dtype=dtype)
                self.Inception_ResNet_B.append(block)
                self.param.append(block.param)
        
        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        self.branch_0_7a = conv2d_bn(self.Inception_ResNet_B[-1].output_size, 256, 1, dtype=dtype)
        self.branch_0_1_7a = conv2d_bn(self.branch_0_7a.output_size, 384, 3)
        self.branch_1_7a = conv2d_bn(self.Inception_ResNet_B[-1].output_size, 256, 1, dtype=dtype)
        self.branch_1_1_7a = conv2d_bn(self.branch_1_7a.output_size, 288, 3, dtype=dtype)
        self.branch_2_7a = conv2d_bn(self.Inception_ResNet_B[-1].output_size, 256, 1, dtype=dtype)
        self.branch_2_1_7a = conv2d_bn(self.branch_2_7a.output_size, 288, 3, dtype=dtype)
        self.branch_2_2_7a = conv2d_bn(self.branch_2_1_7a.output_size, 320, 3, dtype=dtype)
        self.branch_pool_7a = tf.nn.max_pool2d
        self.param.extend([self.branch_0_7a.param,self.branch_0_1_7a.param,self.branch_1_7a.param,
                          self.branch_1_1_7a.param,self.branch_2_7a.param,self.branch_2_1_7a.param,
                          self.branch_2_2_7a.param])
        output_size=self.branch_0_1_7a.output_size+self.branch_1_1_7a.output_size+self.branch_2_2_7a.output_size+self.Inception_ResNet_B[-1].output_size
        
        # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
        self.Inception_ResNet_C=[]
        for i in range(1, 10):
            if i==1:
                block = inception_resnet_block(output_size, scale=0.2, block_type="block8", dtype=dtype)
                self.Inception_ResNet_C.append(block)
                self.param.append(block.param)
            else:
                block = inception_resnet_block(self.Inception_ResNet_C[-1].output_size, scale=0.2, block_type="block8", dtype=dtype)
                self.Inception_ResNet_C.append(block)
                self.param.append(block.param)
        block = inception_resnet_block(self.Inception_ResNet_C[-1].output_size, scale=1.0, activation=None, block_type="block8", dtype=dtype)
        self.Inception_ResNet_C.append(block)
        self.param.append(block.param)
        
        # Final convolution block: 8 x 8 x 1536
        self.conv_7b = conv2d_bn(self.Inception_ResNet_C[-1].output_size, 1536, 1, dtype=dtype)
        self.fc_weight = initializer([self.conv_7b.output_size, self.classes], 'Xavier', dtype)
        self.fc_bias = initializer([self.classes], 'Xavier', dtype)
        self.param.extend([self.conv_7b.param,self.fc_weight,self.fc_bias])
        self.optimizer=Adam(param=self.param)
        return
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                # Stem block: 35 x 35 x 192
                data=self.Stem_layer1.output(data, 2, padding='VALID')
                data=self.Stem_layer2.output(data, padding='VALID')
                data=self.Stem_layer3.output(data)
                data=self.Stem_layer4(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                data=self.Stem_layer5.output(data, padding='VALID')
                data=self.Stem_layer6.output(data, padding='VALID')
                data=self.Stem_layer7(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                
                # Mixed 5b (Inception-A block): 35 x 35 x 320
                data1=self.branch_0_5b.output(data)
                data2=self.branch_1_5b.output(data)
                data2=self.branch_1_1_5b.output(data2)
                data3=self.branch_2_5b.output(data)
                data3=self.branch_2_1_5b.output(data3)
                data3=self.branch_2_2_5b.output(data3)
                data4=self.branch_avg_5b(data, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                data4=self.branch_pool_5b.output(data4)
                branches = [data1, data2, data3, data4]
                data=tf.concat(branches,axis=3)
                
                # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
                for block in self.Inception_ResNet_A:
                    data=block.output(data)
                
                # Mixed 6a (Reduction-A block): 17 x 17 x 1088
                data1=self.branch_0_6a.output(data, 2, padding='VALID')
                data2=self.branch_1_6a.output(data)
                data2=self.branch_1_1_6a.output(data2)
                data2=self.branch_1_2_6a.output(data2, 2, padding='VALID')
                data3=self.branch_pool_6a(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                branches = [data1, data2, data3]
                data=tf.concat(branches,axis=3)
                
                # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
                for block in self.Inception_ResNet_B:
                    data=block.output(data)
                
                # Mixed 7a (Reduction-B block): 8 x 8 x 2080
                data1=self.branch_0_7a.output(data)
                data1=self.branch_0_1_7a.output(data1, 2, padding='VALID')
                data2=self.branch_1_7a.output(data)
                data2=self.branch_1_1_7a.output(data2, 2, padding='VALID')
                data3=self.branch_2_7a.output(data)
                data3=self.branch_2_1_7a.output(data3)
                data3=self.branch_2_2_7a.output(data3, 2, padding='VALID')
                data4=self.branch_pool_7a(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                branches=[data1,data2,data3,data4]
                data=tf.concat(branches,axis=3)
                
                # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
                for block in self.Inception_ResNet_C:
                    data=block.output(data)
                
                # Final convolution block: 8 x 8 x 1536
                data=self.conv_7b.output(data)
                
                if self.include_top:
                    # perform global average pooling on the result
                    data = tf.reduce_mean(data, axis=[1, 2])
                    # perform matrix multiplication with the fully connected weight
                    data = tf.matmul(data, self.fc_weight)+self.fc_bias
                    # apply softmax activation function on the result to get the class probabilities
                    data = tf.nn.softmax(data)
                else:
                    # check the pooling option of the model
                    if self.pooling == 'avg':
                        # perform global average pooling on the result
                        data = tf.reduce_mean(data, axis=[1, 2])
                    elif self.pooling == 'max':
                        # perform global max pooling on the result
                        data = tf.reduce_max(data, axis=[1, 2])
                return data
        else:
            # Stem block: 35 x 35 x 192
            data=self.Stem_layer1.output(data, 2, padding='VALID' ,train_flag=self.km)
            data=self.Stem_layer2.output(data, padding='VALID' ,train_flag=self.km)
            data=self.Stem_layer3.output(data ,train_flag=self.km)
            data=self.Stem_layer4(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            data=self.Stem_layer5.output(data, padding='VALID' ,train_flag=self.km)
            data=self.Stem_layer6.output(data, padding='VALID' ,train_flag=self.km)
            data=self.Stem_layer7(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            # Mixed 5b (Inception-A block): 35 x 35 x 320
            data1=self.branch_0_5b.output(data ,train_flag=self.km)
            data2=self.branch_1_5b.output(data ,train_flag=self.km)
            data2=self.branch_1_1_5b.output(data2 ,train_flag=self.km)
            data3=self.branch_2_5b.output(data ,train_flag=self.km)
            data3=self.branch_2_1_5b.output(data3 ,train_flag=self.km)
            data3=self.branch_2_2_5b.output(data3 ,train_flag=self.km)
            data4=self.branch_avg_5b(data, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            data4=self.branch_pool_5b.output(data4 ,train_flag=self.km)
            branches = [data1, data2, data3, data4]
            data=tf.concat(branches,axis=3)
            
            # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
            for block in self.Inception_ResNet_A:
                data=block.output(data ,train_flag=self.km)
            
            # Mixed 6a (Reduction-A block): 17 x 17 x 1088
            data1=self.branch_0_6a.output(data, 2, padding='VALID' ,train_flag=self.km)
            data2=self.branch_1_6a.output(data ,train_flag=self.km)
            data2=self.branch_1_1_6a.output(data2 ,train_flag=self.km)
            data2=self.branch_1_2_6a.output(data2, 2, padding='VALID' ,train_flag=self.km)
            data3=self.branch_pool_6a(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            branches = [data1, data2, data3]
            data=tf.concat(branches,axis=3)
            
            # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
            for block in self.Inception_ResNet_B:
                data=block.output(data ,train_flag=self.km)
            
            # Mixed 7a (Reduction-B block): 8 x 8 x 2080
            data1=self.branch_0_7a.output(data ,train_flag=self.km)
            data1=self.branch_0_1_7a.output(data1, 2, padding='VALID' ,train_flag=self.km)
            data2=self.branch_1_7a.output(data ,train_flag=self.km)
            data2=self.branch_1_1_7a.output(data2, 2, padding='VALID' ,train_flag=self.km)
            data3=self.branch_2_7a.output(data ,train_flag=self.km)
            data3=self.branch_2_1_7a.output(data3 ,train_flag=self.km)
            data3=self.branch_2_2_7a.output(data3, 2, padding='VALID' ,train_flag=self.km)
            data4=self.branch_pool_7a(data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            branches=[data1,data2,data3,data4]
            data=tf.concat(branches,axis=3)
            
            # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
            for block in self.Inception_ResNet_C:
                data=block.output(data ,train_flag=self.km)
            
            # Final convolution block: 8 x 8 x 1536
            data=self.conv_7b.output(data ,train_flag=self.km)
            
            if self.include_top:
                # perform global average pooling on the result
                data = tf.reduce_mean(data, axis=[1, 2])
                # perform matrix multiplication with the fully connected weight
                data = tf.matmul(data, self.fc_weight)+self.fc_bias
                # apply softmax activation function on the result to get the class probabilities
                data = tf.nn.softmax(data)
            else:
                # check the pooling option of the model
                if self.pooling == 'avg':
                    # perform global average pooling on the result
                    data = tf.reduce_mean(data, axis=[1, 2])
                elif self.pooling == 'max':
                    # perform global max pooling on the result
                    data = tf.reduce_max(data, axis=[1, 2])
            return data
    
    
    # define a method for calculating the loss value
    def loss(self,output,labels,p):
        # assign the device for parallel computation
        with tf.device(assign_device(p,self.device)):
            # calculate the categorical crossentropy loss between output and labels 
            loss=self.loss_object(labels,output)
            # return the loss value    
            return loss
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
            return tape,output,loss
    
    
    # define a method for applying the optimizer
    def opt(self,gradient,p):
        # assign the device for parallel computation
        with tf.device(assign_device(p,self.device)):
            # update the parameters with the gradient and the batch count using the Adam optimizer
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            # return the updated parameters
            return param