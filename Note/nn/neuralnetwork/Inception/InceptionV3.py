import tensorflow as tf
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device


class conv2d_bn:
    def __init__(self,input_channels,filters,num_row,num_col,dtype='float32'):
        self.weight=initializer([num_row,num_col,input_channels,filters],'Xavier',dtype)
        self.activation=activation_dict['relu']
        self.param=[self.weight]
        self.moving_mean=tf.zeros([filters])
        self.moving_var=tf.ones([filters])
        self.beta=tf.Variable(tf.zeros([filters]))
        self.output_size=filters
        self.param.append(self.beta)
    
    
    def output(self,data,strides=(1,1),padding="SAME",train_flag=True):
        data=tf.nn.conv2d(data,self.weight,strides=strides,padding=padding)
        if train_flag:
            mean,var=tf.nn.moments(data,axes=3,keepdims=True)
            self.moving_mean=self.moving_mean*0.99+mean*(1-0.99)
            self.moving_var=self.moving_var*0.99+var*(1-0.99)
            data=tf.nn.batch_normalization(data,mean,var,self.beta,None,1e-3)
        else:
            data=tf.nn.batch_normalization(data,self.moving_mean,self.moving_var,self.beta,None,1e-3)
        output=self.activation(data)
        return output


class inception_block:
    def __init__(self,input_channels,block_type,dtype='float32'):
        if block_type=='17':
            self.branch1x1 = conv2d_bn(input_channels, 192, 1, 1, dtype)
            self.branch7x7_1 = conv2d_bn(input_channels, 160, 1, 1, dtype)
            self.branch7x7_2 = conv2d_bn(self.branch7x7_1.output_size, 160, 1, 7, dtype)
            self.branch7x7_3 = conv2d_bn(self.branch7x7_2.output_size, 192, 7, 1, dtype)
    
            self.branch7x7dbl_1 = conv2d_bn(input_channels, 160, 1, 1, dtype)
            self.branch7x7dbl_2 = conv2d_bn(self.branch7x7dbl_1.output_size, 160, 7, 1, dtype)
            self.branch7x7dbl_3 = conv2d_bn(self.branch7x7dbl_2.output_size, 160, 1, 7, dtype)
            self.branch7x7dbl_4 = conv2d_bn(self.branch7x7dbl_3.output_size, 160, 7, 1, dtype)
            self.branch7x7dbl_5 = conv2d_bn(self.branch7x7dbl_4.output_size, 192, 1, 7, dtype)
            self.avgpool2d = tf.nn.avg_pool2d
            self.branch_pool = conv2d_bn(input_channels, 192, 1, 1, dtype)
            self.output_size=self.branch1x1.output_size+self.branch7x7_3.output_size+self.branch7x7dbl_5.output_size+self.branch_pool.output_size
            self.param=[self.branch1x1.param,self.branch7x7_1.param,self.branch7x7_2.param,self.branch7x7_3.param,
                        self.branch7x7dbl_1.param,self.branch7x7dbl_2.param,self.branch7x7dbl_3.param,self.branch7x7dbl_4.param,
                        self.branch7x7dbl_5.param,self.branch_pool.param]
        elif block_type=='8':
            self.branch1x1 = conv2d_bn(input_channels, 320, 1, 1, dtype)

            self.branch3x3 = conv2d_bn(input_channels, 384, 1, 1, dtype)
            self.branch3x3_1 = conv2d_bn(self.branch3x3.output_size, 384, 1, 3, dtype)
            self.branch3x3_2 = conv2d_bn(self.branch3x3.output_size, 384, 3, 1, dtype)
    
            self.branch3x3dbl_1_ = conv2d_bn(input_channels, 448, 1, 1, dtype)
            self.branch3x3dbl_2_ = conv2d_bn(self.branch3x3dbl_1_.output_size, 384, 3, 3, dtype)
            self.branch3x3dbl_1 = conv2d_bn(self.branch3x3dbl_2_.output_size, 384, 1, 3, dtype)
            self.branch3x3dbl_2 = conv2d_bn(self.branch3x3dbl_2_.output_size, 384, 3, 1, dtype)
    
            self.avgpool2d = tf.nn.avg_pool2d
            self.branch_pool = conv2d_bn(input_channels, 192, 1, 1, dtype)
            self.param=[self.branch1x1.param,self.branch3x3.param,self.branch3x3_1.param,self.branch3x3_2.param,
                        self.branch3x3dbl_1_.param,self.branch3x3dbl_2_.param,self.branch3x3dbl_1.param,
                        self.branch3x3dbl_2.param,self.branch_pool.param]
            self.output_size=self.branch1x1.output_size+self.branch3x3_1.output_size+self.branch3x3_2.output_size+self.branch3x3dbl_1.output_size+self.branch3x3dbl_2.output_size+self.branch_pool.output_size
        self.block_type=block_type
    
    
    def output(self,data,train_flag=True):
        if self.block_type=='17':
            data1=self.branch1x1.output(data)
            data2=self.branch7x7_1.output(data)
            data2=self.branch7x7_2.output(data2)
            data2=self.branch7x7_3.output(data2)
            
            data3=self.branch7x7dbl_1.output(data)
            data3=self.branch7x7dbl_2.output(data3)
            data3=self.branch7x7dbl_3.output(data3)
            data3=self.branch7x7dbl_4.output(data3)
            data3=self.branch7x7dbl_5.output(data3)
            data4=self.avgpool2d(data, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
            data4=self.branch_pool.output(data4)
            branches=[data1,data2,data3,data4]
        elif self.block_type=='8':
            data1=self.branch1x1.output(data)
            data2=self.branch3x3.output(data)
            data2_1=self.branch3x3_1.output(data2)
            data2_2=self.branch3x3_2.output(data2)
            data2=tf.concat([data2_1,data2_2],axis=3)
            data3=self.branch3x3dbl_1_.output(data)
            data3=self.branch3x3dbl_2_.output(data3)
            data3_1=self.branch3x3dbl_1.output(data3)
            data3_2=self.branch3x3dbl_2.output(data3)
            data3=tf.concat([data3_1,data3_2],axis=3)
            data4=self.avgpool2d(data, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
            data4=self.branch_pool.output(data4)
            branches=[data1,data2,data3,data4]
        return tf.concat(branches,axis=3)
            

class InceptionV3:
    def __init__(self,classes=1000,include_top=True,pooling=None,device='GPU'):
        self.classes=classes
        self.include_top=include_top
        self.pooling=pooling
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer=Adam()
        self.km=0
    
    
    def build(self,dtype='float32'):
        self.conv2d_bn1 = conv2d_bn(3, 32, 3, 3, dtype)
        self.conv2d_bn2 = conv2d_bn(self.conv2d_bn1.output_size, 32, 3, 3, dtype)
        self.conv2d_bn3 = conv2d_bn(self.conv2d_bn2.output_size, 64, 3, 3, dtype)
        self.maxpool2d1 = tf.nn.max_pool2d
    
        self.conv2d_bn4 = conv2d_bn(self.conv2d_bn3.output_size, 80, 1, 1, dtype)
        self.conv2d_bn5 = conv2d_bn(self.conv2d_bn4.output_size, 192, 3, 3, dtype)
        self.maxpool2d2 = tf.nn.max_pool2d
        self.param=[self.conv2d_bn1.param,self.conv2d_bn2.param,self.conv2d_bn3.param,self.conv2d_bn4.param,
                    self.conv2d_bn5.param]
        
        # mixed 0: 35 x 35 x 256
        self.branch1x1_1 = conv2d_bn(self.conv2d_bn5.output_size, 64, 1, 1, dtype)

        self.branch5x5_1 = conv2d_bn(self.conv2d_bn5.output_size, 48, 1, 1, dtype)
        self.branch5x5_2 = conv2d_bn(self.branch5x5_1.output_size, 64, 5, 5, dtype)
    
        self.branch3x3dbl_1 = conv2d_bn(self.conv2d_bn5.output_size, 64, 1, 1, dtype)
        self.branch3x3dbl_2 = conv2d_bn(self.branch3x3dbl_1.output_size, 96, 3, 3, dtype)
        self.branch3x3dbl_3 = conv2d_bn(self.branch3x3dbl_2.output_size, 96, 3, 3, dtype)
    
        self.avgpool2d1 = tf.nn.avg_pool2d
        self.branch_pool1 = conv2d_bn(self.conv2d_bn5.output_size, 32, 1, 1, dtype)
        output_size=self.branch1x1_1.output_size+self.branch5x5_2.output_size+self.branch3x3dbl_3.output_size+self.branch_pool1.output_size
        self.param.extend([self.branch1x1_1.param,self.branch5x5_1.param,self.branch5x5_2.param,self.branch3x3dbl_1.param,
                           self.branch3x3dbl_2.param,self.branch3x3dbl_3.param,self.branch_pool1.param])
        
        # mixed 1: 35 x 35 x 288
        self.branch1x1_2 = conv2d_bn(output_size, 64, 1, 1, dtype)

        self.branch5x5_3 = conv2d_bn(output_size, 48, 1, 1, dtype)
        self.branch5x5_4 = conv2d_bn(self.branch5x5_3.output_size, 64, 5, 5, dtype)
    
        self.branch3x3dbl_4 = conv2d_bn(output_size, 64, 1, 1, dtype)
        self.branch3x3dbl_5 = conv2d_bn(self.branch3x3dbl_4.output_size, 96, 3, 3, dtype)
        self.branch3x3dbl_6 = conv2d_bn(self.branch3x3dbl_5.output_size, 96, 3, 3, dtype)
    
        self.avgpool2d2 = tf.nn.avg_pool2d
        self.branch_pool2 = conv2d_bn(output_size, 64, 1, 1, dtype)
        output_size=self.branch1x1_2.output_size+self.branch5x5_4.output_size+self.branch3x3dbl_6.output_size+self.branch_pool2.output_size
        self.param.extend([self.branch1x1_2.param,self.branch5x5_3.param,self.branch5x5_4.param,self.branch3x3dbl_4.param,
                           self.branch3x3dbl_5.param,self.branch3x3dbl_6.param,self.branch_pool2.param])
        
        # mixed 2: 35 x 35 x 288
        self.branch1x1_3 = conv2d_bn(output_size, 64, 1, 1, dtype)

        self.branch5x5_5 = conv2d_bn(output_size, 48, 1, 1, dtype)
        self.branch5x5_6 = conv2d_bn(self.branch5x5_5.output_size, 64, 5, 5, dtype)
    
        self.branch3x3dbl_7 = conv2d_bn(output_size, 64, 1, 1, dtype)
        self.branch3x3dbl_8 = conv2d_bn(self.branch3x3dbl_7.output_size, 96, 3, 3, dtype)
        self.branch3x3dbl_9 = conv2d_bn(self.branch3x3dbl_8.output_size, 96, 3, 3, dtype)
    
        self.avgpool2d3 = tf.nn.avg_pool2d
        self.branch_pool3 = conv2d_bn(output_size, 64, 1, 1, dtype)
        output_size=self.branch1x1_3.output_size+self.branch5x5_6.output_size+self.branch3x3dbl_9.output_size+self.branch_pool3.output_size
        self.param.extend([self.branch1x1_3.param,self.branch5x5_5.param,self.branch5x5_6.param,self.branch3x3dbl_7.param,
                           self.branch3x3dbl_8.param,self.branch3x3dbl_9.param,self.branch_pool3.param])
        
        # mixed 3: 17 x 17 x 768
        self.branch3x3_1 = conv2d_bn(output_size, 384, 3, 3, dtype)

        self.branch3x3dbl_10 = conv2d_bn(output_size, 64, 1, 1, dtype)
        self.branch3x3dbl_11 = conv2d_bn(self.branch3x3dbl_10.output_size, 96, 3, 3, dtype)
        self.branch3x3dbl_12 = conv2d_bn(self.branch3x3dbl_11.output_size, 96, 3, 3, dtype)
    
        self.maxpool2d3 = tf.nn.max_pool2d
        output_size=self.branch3x3_1.output_size+self.branch3x3dbl_12.output_size+output_size
        self.param.extend([self.branch3x3_1.param,self.branch3x3dbl_10.param,self.branch3x3dbl_11.param,
                           self.branch3x3dbl_12.param])

        # mixed 4: 17 x 17 x 768
        self.branch1x1_4 = conv2d_bn(output_size, 192, 1, 1, dtype)

        self.branch7x7_1 = conv2d_bn(output_size, 128, 1, 1, dtype)
        self.branch7x7_2 = conv2d_bn(self.branch7x7_1.output_size, 128, 1, 7, dtype)
        self.branch7x7_3 = conv2d_bn(self.branch7x7_2.output_size, 192, 7, 1, dtype)
    
        self.branch7x7dbl_1 = conv2d_bn(output_size, 128, 1, 1, dtype)
        self.branch7x7dbl_2 = conv2d_bn(self.branch7x7dbl_1.output_size, 128, 7, 1, dtype)
        self.branch7x7dbl_3 = conv2d_bn(self.branch7x7dbl_2.output_size, 128, 1, 7, dtype)
        self.branch7x7dbl_4 = conv2d_bn(self.branch7x7dbl_3.output_size, 128, 7, 1, dtype)
        self.branch7x7dbl_5 = conv2d_bn(self.branch7x7dbl_4.output_size, 192, 1, 7, dtype)
    
        self.avgpool2d4 = tf.nn.avg_pool2d
        self.branch_pool5 = conv2d_bn(output_size, 192, 1, 1, dtype)
        output_size=self.branch1x1_4.output_size+self.branch7x7_3.output_size+self.branch7x7dbl_5.output_size+self.branch_pool5.output_size
        self.param.extend([self.branch1x1_4.param,self.branch7x7_1.param,self.branch7x7_2.param,self.branch7x7_3.param,
                           self.branch7x7dbl_1.param,self.branch7x7dbl_2.param,self.branch7x7dbl_3.param,
                           self.branch7x7dbl_4.param,self.branch7x7dbl_5.param,self.branch_pool5.param])

        # mixed 5, 6: 17 x 17 x 768
        self.blocks1=[]
        for i in range(2):
            block = inception_block(output_size, "17", dtype)
            self.blocks1.append(block)
            output_size=block.output_size
            self.param.extend([block.param])
        
        # mixed 7: 17 x 17 x 768
        self.branch1x1_5 = conv2d_bn(output_size, 192, 1, 1, dtype)

        self.branch7x7_4 = conv2d_bn(output_size, 192, 1, 1, dtype)
        self.branch7x7_5 = conv2d_bn(self.branch7x7_4.output_size, 192, 1, 7, dtype)
        self.branch7x7_6 = conv2d_bn(self.branch7x7_5.output_size, 192, 7, 1, dtype)
    
        self.branch7x7dbl_6 = conv2d_bn(output_size, 192, 1, 1)
        self.branch7x7dbl_7 = conv2d_bn(self.branch7x7dbl_6.output_size, 192, 7, 1, dtype)
        self.branch7x7dbl_8 = conv2d_bn(self.branch7x7dbl_7.output_size, 192, 1, 7, dtype)
        self.branch7x7dbl_9 = conv2d_bn(self.branch7x7dbl_8.output_size, 192, 7, 1, dtype)
        self.branch7x7dbl_10 = conv2d_bn(self.branch7x7dbl_9.output_size, 192, 1, 7, dtype)
    
        self.avgpool2d5 = tf.nn.avg_pool2d
        self.branch_pool6 = conv2d_bn(output_size, 192, 1, 1, dtype=dtype)
        output_size=self.branch1x1_5.output_size+self.branch7x7_6.output_size+self.branch7x7dbl_10.output_size+self.branch_pool6.output_size
        self.param.extend([self.branch1x1_5.param,self.branch7x7_4.param,self.branch7x7_5.param,self.branch7x7_6.param,
                          self.branch7x7dbl_6.param,self.branch7x7dbl_7.param,self.branch7x7dbl_8.param,self.branch7x7dbl_9.param,
                          self.branch7x7dbl_10.param,self.branch_pool6.param])
        
        # mixed 8: 8 x 8 x 1280
        self.branch3x3_2 = conv2d_bn(output_size, 192, 1, 1, dtype)
        self.branch3x3_3 = conv2d_bn(self.branch3x3_2.output_size, 320, 3, 3, dtype)
    
        self.branch7x7x3_1 = conv2d_bn(output_size, 192, 1, 1)
        self.branch7x7x3_2 = conv2d_bn(self.branch7x7x3_1.output_size, 192, 1, 7, dtype)
        self.branch7x7x3_3 = conv2d_bn(self.branch7x7x3_2.output_size, 192, 7, 1, dtype)
        self.branch7x7x3_4 = conv2d_bn(self.branch7x7x3_3.output_size, 192, 3, 3, dtype)
    
        self.maxpool2d4 = tf.nn.max_pool2d
        output_size=self.branch3x3_3.output_size+self.branch7x7x3_4.output_size+output_size
        self.param.extend([self.branch3x3_2.param,self.branch3x3_3.param,self.branch7x7x3_1.param,self.branch7x7x3_2.param,
                           self.branch7x7x3_3.param,self.branch7x7x3_4.param])
        
        # mixed 9: 8 x 8 x 2048
        self.blocks2=[]
        for i in range(2):
            block = inception_block(output_size, "8", dtype)
            self.blocks2.append(block)
            output_size=block.output_size
            self.param.extend([block.param])
        self.fc_weight = initializer([output_size, self.classes], 'Xavier', dtype)
        self.fc_bias = initializer([self.classes], 'Xavier', dtype)
        self.param.extend([self.fc_weight,self.fc_bias])
        self.optimizer=Adam(param=self.param)
        return
    
    
    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x=self.conv2d_bn1.output(data,(2,2),'VALID')
                x=self.conv2d_bn2.output(x,padding='VALID')
                x=self.conv2d_bn3.output(x)
                x=self.maxpool2d1(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
            
                x=self.conv2d_bn4.output(x,padding='VALID')
                x=self.conv2d_bn5.output(x,padding='VALID')
                x=self.maxpool2d2(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
                
                # mixed 0: 35 x 35 x 256
                branch1x1=self.branch1x1_1.output(x)

                branch5x5=self.branch5x5_1.output(x)
                branch5x5=self.branch5x5_2.output(branch5x5)
            
                branch3x3dbl=self.branch3x3dbl_1.output(x)
                branch3x3dbl=self.branch3x3dbl_2.output(branch3x3dbl)
                branch3x3dbl=self.branch3x3dbl_3.output(branch3x3dbl)
            
                branch_pool=self.avgpool2d1(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
                branch_pool=self.branch_pool1.output(branch_pool)
                x=tf.concat([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3)
                
                # mixed 1: 35 x 35 x 288
                branch1x1=self.branch1x1_2.output(x)

                branch5x5=self.branch5x5_3.output(x)
                branch5x5=self.branch5x5_4.output(branch5x5)
            
                branch3x3dbl=self.branch3x3dbl_4.output(x)
                branch3x3dbl=self.branch3x3dbl_5.output(branch3x3dbl)
                branch3x3dbl=self.branch3x3dbl_6.output(branch3x3dbl)
            
                branch_pool=self.avgpool2d2(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
                branch_pool=self.branch_pool2.output(branch_pool)
                x=tf.concat([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3)
                
                # mixed 2: 35 x 35 x 288
                branch1x1=self.branch1x1_3.output(x)

                branch5x5=self.branch5x5_5.output(x)
                branch5x5=self.branch5x5_6.output(branch5x5)
            
                branch3x3dbl=self.branch3x3dbl_7.output(x)
                branch3x3dbl=self.branch3x3dbl_8.output(branch3x3dbl)
                branch3x3dbl=self.branch3x3dbl_9.output(branch3x3dbl)
            
                branch_pool=self.avgpool2d3(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
                branch_pool=self.branch_pool3.output(branch_pool)
                x=tf.concat([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3)
                
                # mixed 3: 17 x 17 x 768
                branch3x3=self.branch3x3_1.output(x,(2,2),padding='VALID')

                branch3x3dbl=self.branch3x3dbl_10.output(x)
                branch3x3dbl=self.branch3x3dbl_11.output(branch3x3dbl)
                branch3x3dbl=self.branch3x3dbl_12.output(branch3x3dbl,(2,2),padding='VALID')
    
                branch_pool=self.maxpool2d3(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
                x=tf.concat([branch3x3,branch3x3dbl,branch_pool],axis=3)
                
                # mixed 4: 17 x 17 x 768
                branch1x1=self.branch1x1_4.output(x)

                branch7x7=self.branch7x7_1.output(x)
                branch7x7=self.branch7x7_2.output(branch7x7)
                branch7x7=self.branch7x7_3.output(branch7x7)
            
                branch7x7dbl=self.branch7x7dbl_1.output(x)
                branch7x7dbl=self.branch7x7dbl_2.output(branch7x7dbl)
                branch7x7dbl=self.branch7x7dbl_3.output(branch7x7dbl)
                branch7x7dbl=self.branch7x7dbl_4.output(branch7x7dbl)
                branch7x7dbl=self.branch7x7dbl_5.output(branch7x7dbl)
            
                branch_pool=self.avgpool2d4(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
                branch_pool=self.branch_pool5.output(branch_pool)
                x=tf.concat([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3)
                
                # mixed 5, 6: 17 x 17 x 768
                for block in self.blocks1:
                    x=block.output(x)
                
                # mixed 7: 17 x 17 x 768
                branch1x1=self.branch1x1_4.output(x)

                branch7x7=self.branch7x7_4.output(x)
                branch7x7=self.branch7x7_5.output(branch7x7)
                branch7x7=self.branch7x7_6.output(branch7x7)
            
                branch7x7dbl=self.branch7x7dbl_6.output(x)
                branch7x7dbl=self.branch7x7dbl_7.output(branch7x7dbl)
                branch7x7dbl=self.branch7x7dbl_8.output(branch7x7dbl)
                branch7x7dbl=self.branch7x7dbl_9.output(branch7x7dbl)
                branch7x7dbl=self.branch7x7dbl_10.output(branch7x7dbl)
            
                branch_pool=self.avgpool2d5(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
                branch_pool=self.branch_pool6.output(branch_pool)
                x=tf.concat([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3)
                
                # mixed 8: 8 x 8 x 1280
                branch3x3=self.branch3x3_2.output(x)
                branch3x3=self.branch3x3_3.output(branch3x3,(2,2),'VALID')
            
                branch7x7x3=self.branch7x7x3_1.output(x)
                branch7x7x3=self.branch7x7x3_2.output(branch7x7x3)
                branch7x7x3=self.branch7x7x3_3.output(branch7x7x3)
                branch7x7x3=self.branch7x7x3_4.output(branch7x7x3,(2,2),'VALID')
            
                branch_pool=self.maxpool2d4(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
                x=tf.concat([branch3x3,branch7x7x3,branch_pool],axis=3)
                
                # mixed 9: 8 x 8 x 2048
                for block in self.blocks2:
                    x=block.output(x)
                    
                if self.include_top:
                    data = tf.reduce_mean(x, axis=[1, 2])
                    data = tf.matmul(data, self.fc_weight)+self.fc_bias
                    data = tf.nn.softmax(data)
                else:
                    if self.pooling == 'avg':
                        data = tf.reduce_mean(x, axis=[1, 2])
                    elif self.pooling == 'max':
                        data = tf.reduce_max(x, axis=[1, 2])
                return data
        else:
            x=self.conv2d_bn1.output(data,(2,2),'VALID',train_flag=self.km)
            x=self.conv2d_bn2.output(x,padding='VALID',train_flag=self.km)
            x=self.conv2d_bn3.output(x,train_flag=self.km)
            x=self.maxpool2d1(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
        
            x=self.conv2d_bn4.output(x,padding='VALID',train_flag=self.km)
            x=self.conv2d_bn5.output(x,padding='VALID',train_flag=self.km)
            x=self.maxpool2d2(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
            
            # mixed 0: 35 x 35 x 256
            branch1x1=self.branch1x1_1.output(x,train_flag=self.km)

            branch5x5=self.branch5x5_1.output(x,train_flag=self.km)
            branch5x5=self.branch5x5_2.output(branch5x5,train_flag=self.km)
        
            branch3x3dbl=self.branch3x3dbl_1.output(x,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_2.output(branch3x3dbl,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_3.output(branch3x3dbl,train_flag=self.km)
        
            branch_pool=self.avgpool2d1(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
            branch_pool=self.branch_pool1.output(branch_pool,train_flag=self.km)
            x=tf.concat([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3)
            
            # mixed 1: 35 x 35 x 288
            branch1x1=self.branch1x1_2.output(x,train_flag=self.km)

            branch5x5=self.branch5x5_3.output(x,train_flag=self.km)
            branch5x5=self.branch5x5_4.output(branch5x5,train_flag=self.km)
        
            branch3x3dbl=self.branch3x3dbl_4.output(x,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_5.output(branch3x3dbl,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_6.output(branch3x3dbl,train_flag=self.km)
        
            branch_pool=self.avgpool2d2(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
            branch_pool=self.branch_pool2.output(branch_pool,train_flag=self.km)
            x=tf.concat([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3)
            
            # mixed 2: 35 x 35 x 288
            branch1x1=self.branch1x1_3.output(x,train_flag=self.km)

            branch5x5=self.branch5x5_5.output(x,train_flag=self.km)
            branch5x5=self.branch5x5_6.output(branch5x5,train_flag=self.km)
        
            branch3x3dbl=self.branch3x3dbl_7.output(x,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_8.output(branch3x3dbl,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_9.output(branch3x3dbl,train_flag=self.km)
        
            branch_pool=self.avgpool2d3(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
            branch_pool=self.branch_pool3.output(branch_pool,train_flag=self.km)
            x=tf.concat([branch1x1,branch5x5,branch3x3dbl,branch_pool],axis=3)
            
            # mixed 3: 17 x 17 x 768
            branch3x3=self.branch3x3_1.output(x,(2,2),padding='VALID',train_flag=self.km)

            branch3x3dbl=self.branch3x3dbl_10.output(x,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_11.output(branch3x3dbl,train_flag=self.km)
            branch3x3dbl=self.branch3x3dbl_12.output(branch3x3dbl,(2,2),padding='VALID',train_flag=self.km)

            branch_pool=self.maxpool2d3(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
            x=tf.concat([branch3x3,branch3x3dbl,branch_pool],axis=3)
            
            # mixed 4: 17 x 17 x 768
            branch1x1=self.branch1x1_4.output(x,train_flag=self.km)

            branch7x7=self.branch7x7_1.output(x,train_flag=self.km)
            branch7x7=self.branch7x7_2.output(branch7x7,train_flag=self.km)
            branch7x7=self.branch7x7_3.output(branch7x7,train_flag=self.km)
        
            branch7x7dbl=self.branch7x7dbl_1.output(x,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_2.output(branch7x7dbl,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_3.output(branch7x7dbl,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_4.output(branch7x7dbl,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_5.output(branch7x7dbl,train_flag=self.km)
        
            branch_pool=self.avgpool2d4(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
            branch_pool=self.branch_pool5.output(branch_pool,train_flag=self.km)
            x=tf.concat([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3)
            
            # mixed 5, 6: 17 x 17 x 768
            for block in self.blocks1:
                x=block.output(x,train_flag=self.km)
            
            # mixed 7: 17 x 17 x 768
            branch1x1=self.branch1x1_4.output(x,train_flag=self.km)

            branch7x7=self.branch7x7_4.output(x,train_flag=self.km)
            branch7x7=self.branch7x7_5.output(branch7x7,train_flag=self.km)
            branch7x7=self.branch7x7_6.output(branch7x7,train_flag=self.km)
        
            branch7x7dbl=self.branch7x7dbl_6.output(x,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_7.output(branch7x7dbl,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_8.output(branch7x7dbl,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_9.output(branch7x7dbl,train_flag=self.km)
            branch7x7dbl=self.branch7x7dbl_10.output(branch7x7dbl,train_flag=self.km)
        
            branch_pool=self.avgpool2d5(x, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
            branch_pool=self.branch_pool6.output(branch_pool,train_flag=self.km)
            x=tf.concat([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3)
            
            # mixed 8: 8 x 8 x 1280
            branch3x3=self.branch3x3_2.output(x,train_flag=self.km)
            branch3x3=self.branch3x3_3.output(branch3x3,(2,2),'VALID',train_flag=self.km)
        
            branch7x7x3=self.branch7x7x3_1.output(x,train_flag=self.km)
            branch7x7x3=self.branch7x7x3_2.output(branch7x7x3,train_flag=self.km)
            branch7x7x3=self.branch7x7x3_3.output(branch7x7x3,train_flag=self.km)
            branch7x7x3=self.branch7x7x3_4.output(branch7x7x3,(2,2),'VALID',train_flag=self.km)
        
            branch_pool=self.maxpool2d4(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')
            x=tf.concat([branch3x3,branch7x7x3,branch_pool],axis=3)
            
            # mixed 9: 8 x 8 x 2048
            for block in self.blocks2:
                x=block.output(x,train_flag=self.km)
                
            if self.include_top:
                data = tf.reduce_mean(x, axis=[1, 2])
                data = tf.matmul(data, self.fc_weight)+self.fc_bias
                data = tf.nn.softmax(data)
            else:
                if self.pooling == 'avg':
                    data = tf.reduce_mean(x, axis=[1, 2])
                elif self.pooling == 'max':
                    data = tf.reduce_max(x, axis=[1, 2])
            return data
    
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,self.device)):
            loss=self.loss_object(labels,output)
            return loss
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
            return tape,output,loss
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,self.device)):
            param=self.optimizer.opt(gradient,self.param,self.bc[0])
            return param