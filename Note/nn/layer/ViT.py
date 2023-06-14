import tensorflow as tf
from Note.nn.layer.self_attention import self_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization
from Note.nn.activation import activation_dict


class ViT:
    def __init__(self,dim,num_heads,mlp_dim,activation='gelu',dropout=0.1):
        self.norm1=layer_normalization(dim)
        self.attn=self_attention([dim,dim])
        self.attn_dropout=tf.keras.layers.Dropout(dropout)
        self.norm2=layer_normalization(dim)
        self.mlp=dense([dim,mlp_dim,dim],activation=activation)
        self.mlp_dropout=tf.keras.layers.Dropout(dropout)
        self.num_heads=num_heads
        self.activation=activation_dict[activation]
    
    
    def output(self,data):
        # data: (B, N, dim)
        attn_input=self.norm1(data) # (B,N,dim)
        attn_output,_=self.attn.output(attn_input,self.num_heads) # (B,N,dim)
        attn_output=self.attn_dropout(attn_output) # (B,N,dim)
        residual_input=data+attn_output # (B,N,dim)
        mlp_input=self.norm2(residual_input) # (B,N,dim)
        mlp_output=self.activation(self.mlp.output(mlp_input)) # (B,N,dim)
        mlp_output=self.mlp_dropout(mlp_output) # (B,N,dim)
        output=residual_input+mlp_output # (B,N,dim)
        return output