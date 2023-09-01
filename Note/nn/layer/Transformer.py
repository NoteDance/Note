import tensorflow as tf
import Note.nn.initializer as i
from Note.nn.layer.layer_normalization import layer_normalization


class Transformer:
    def __init__(self,output_size,num_heads,input_size=None,weight_initializer='Xavier',bias_initializer='zeros',use_bias=True,dtype='float32'):
        self.num_heads=num_heads # number of attention heads
        self.input_size=input_size
        self.weight_initializer=weight_initializer
        self.bias_initializer=bias_initializer
        self.use_bias=use_bias
        self.dtype=dtype
        self.head_size=output_size//num_heads # size of each attention head
        self.train_flag=True
        self.output_size=output_size
        if input_size!=None:
            self.weight_q=i.initializer([input_size,output_size],weight_initializer,dtype) # query weight matrix
            self.weight_k=i.initializer([input_size,output_size],weight_initializer,dtype) # key weight matrix
            self.weight_v=i.initializer([input_size,output_size],weight_initializer,dtype) # value weight matrix
            self.weight_o=i.initializer([output_size,output_size],weight_initializer,dtype) # output weight matrix
            self.weight_ffn_1=i.initializer([output_size,4*output_size],weight_initializer,dtype) # first feed-forward weight matrix
            self.weight_ffn_2=i.initializer([4*output_size,output_size],weight_initializer,dtype) # second feed-forward weight matrix
            if use_bias:
                self.bias_q=i.initializer([output_size],bias_initializer,dtype) # query bias vector
                self.bias_k=i.initializer([output_size],bias_initializer,dtype) # key bias vector
                self.bias_v=i.initializer([output_size],bias_initializer,dtype) # value bias vector
                self.bias_o=i.initializer([output_size],bias_initializer,dtype) # output bias vector
                self.bias_ffn_1=i.initializer([4*output_size],bias_initializer,dtype) # first feed-forward bias vector
                self.bias_ffn_2=i.initializer([output_size],bias_initializer,dtype) # second feed-forward bias vector
            if use_bias:
                self.param=[self.weight_q,self.weight_k,self.weight_v,self.weight_o,
                                  self.weight_ffn_1,self.weight_ffn_2,
                                  self.bias_q,self.bias_k,self.bias_v,self.bias_o,
                                  self.bias_ffn_1,self.bias_ffn_2]
            else:
                self.param=[self.weight_q,self.weight_k,self.weight_v,self.weight_o,
                                  self.weight_ffn_1,self.weight_ffn_2]
    
    
    def build(self):
        self.weight_q=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # query weight matrix
        self.weight_k=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # key weight matrix
        self.weight_v=i.initializer([self.input_size,self.output_size],self.weight_initializer,self.dtype) # value weight matrix
        self.weight_o=i.initializer([self.output_size,self.output_size],self.weight_initializer,self.dtype) # output weight matrix
        self.weight_ffn_1=i.initializer([self.output_size,4*self.output_size],self.weight_initializer,self.dtype) # first feed-forward weight matrix
        self.weight_ffn_2=i.initializer([4*self.output_size,self.output_size],self.weight_initializer,self.dtype) # second feed-forward weight matrix
        if self.use_bias:
            self.bias_q=i.initializer([self.output_size],self.bias_initializer,self.dtype) # query bias vector
            self.bias_k=i.initializer([self.output_size],self.bias_initializer,self.dtype) # key bias vector
            self.bias_v=i.initializer([self.output_size],self.bias_initializer,self.dtype) # value bias vector
            self.bias_o=i.initializer([self.output_size],self.bias_initializer,self.dtype) # output bias vector
            self.bias_ffn_1=i.initializer([4*self.output_size],self.bias_initializer,self.dtype) # first feed-forward bias vector
            self.bias_ffn_2=i.initializer([self.output_size],self.bias_initializer,self.dtype) # second feed-forward bias vector
        if self.use_bias:
            self.param=[self.weight_q,self.weight_k,self.weight_v,self.weight_o,
                              self.weight_ffn_1,self.weight_ffn_2,
                              self.bias_q,self.bias_k,self.bias_v,self.bias_o,
                              self.bias_ffn_1,self.bias_ffn_2]
        else:
            self.param=[self.weight_q,self.weight_k,self.weight_v,self.weight_o,
                              self.weight_ffn_1,self.weight_ffn_2]
        return
    
    
    def output(self,data,train_flag=True):
        self.train_flag=train_flag
        # Compute the query, key and value vectors by applying linear transformation and optional bias
        if self.use_bias:
            q=tf.matmul(data,self.weight_q)+self.bias_q # shape: (batch_size ,seq_len ,hidden_size)
            k=tf.matmul(data,self.weight_k)+self.bias_k# shape: (batch_size ,seq_len ,hidden_size)
            v=tf.matmul(data,self.weight_v)+self.bias_v# shape: (batch_size ,seq_len ,hidden_size)
        else:
            q=tf.matmul(data,self.weight_q)# shape: (batch_size ,seq_len ,hidden_size)
            k=tf.matmul(data,self.weight_k)# shape: (batch_size ,seq_len ,hidden_size)
            v=tf.matmul(data,self.weight_v)# shape: (batch_size ,seq_len ,hidden_size)
        # Reshape the query, key and value vectors to split them into multiple heads
        q=tf.reshape(q,(tf.shape(q)[0],tf.shape(q)[1],self.num_heads,q.shape[-1]//self.num_heads))# shape: (batch_size ,seq_len ,num_heads ,head_size)
        k=tf.reshape(k,(tf.shape(k)[0],tf.shape(k)[1],self.num_heads,k.shape[-1]//self.num_heads))# shape: (batch_size ,seq_len ,num_heads ,head_size)
        v=tf.reshape(v,(tf.shape(v)[0],tf.shape(v)[1],self.num_heads,v.shape[-1]//self.num_heads))# shape: (batch_size ,seq_len ,num_heads ,head_size)
        # Transpose the query, key and value vectors to bring the num_heads dimension to the front
        q=tf.transpose(q,perm=[0,2,1,3])# shape: (batch_size, num_heads, seq_len, head_size)
        k=tf.transpose(k,perm=[0,2,1,3])# shape: (batch_size, num_heads, seq_len, head_size)
        v=tf.transpose(v,perm=[0,2,1,3])# shape: (batch_size, num_heads, seq_len, head_size)
        # Compute the attention weights by scaled dot-product attention
        w=tf.matmul(q,k,transpose_b=True)/tf.math.sqrt(tf.cast(self.head_size,dtype=self.dtype))# shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        w=tf.nn.softmax(w)# shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        # Compute the attention output by multiplying the attention weights and the value vectors
        o=tf.matmul(w,v)# shape: (batch_size, num_heads, seq_len_q, head_size)
        # Transpose and reshape the attention output to concatenate the multiple heads
        o=tf.transpose(o,perm=[0,2,1,3])# shape: (batch_size ,seq_len_q ,num_heads ,head_size)
        o=tf.reshape(o,(tf.shape(o)[0],tf.shape(o)[1],o.shape[2]*o.shape[3]))# shape: (batch_size ,seq_len_q ,hidden_size)
        # Apply another linear transformation and optional bias to get the final output of the multi-head attention sublayer
        if self.use_bias:
          output=tf.matmul(o,self.weight_o)+self.bias_o# shape: (batch_size ,seq_len_q ,hidden_size)
        else:
          output=tf.matmul(o,self.weight_o)# shape: (batch_size ,seq_len_q ,hidden_size)
        # Add residual connection and layer normalization
        output=layer_normalization(output+data,train_flag=self.train_flag) # shape: (batch_size ,seq_len_q ,hidden_size)
        # Apply the first feed-forward sublayer with ReLU activation
        if self.use_bias:
          ffn_1=tf.nn.relu(tf.matmul(output,self.weight_ffn_1)+self.bias_ffn_1)# shape: (batch_size ,seq_len_q ,4 * hidden_size)
        else:
          ffn_1=tf.nn.relu(tf.matmul(output,self.weight_ffn_1))# shape: (batch_size ,seq_len_q ,4 * hidden_size)
        # Apply layer normalization
        ffn_1=layer_normalization(ffn_1,train_flag=self.train_flag) # shape: (batch_size ,seq_len_q ,4 * hidden_size)
        # Apply the second feed-forward sublayer with linear activation
        if self.use_bias:
          ffn_2=tf.matmul(ffn_1,self.weight_ffn_2)+self.bias_ffn_2# shape: (batch_size ,seq_len_q ,hidden_size)
        else:
          ffn_2=tf.matmul(ffn_1,self.weight_ffn_2)# shape: (batch_size ,seq_len_q ,hidden_size)
        # Add residual connection and layer normalization
        output=layer_normalization(ffn_2+output,train_flag=self.train_flag)# shape: (batch_size ,seq_len_q ,hidden_size)
        return output
