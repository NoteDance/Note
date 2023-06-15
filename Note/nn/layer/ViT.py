import tensorflow as tf
from Note.nn.layer.self_attention import self_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_normalization import layer_normalization
from Note.nn.activation import activation_dict


class ViT:
    def __init__(self,dim,num_heads,mlp_dim,activation='gelu',dropout=0.1):
        self.attn=self_attention([dim,dim])
        self.mlp1=dense([dim,mlp_dim])
        self.mlp2=dense([mlp_dim,dim])
        self.num_heads=num_heads
        self.activation=activation_dict[activation]
        self.dropout=dropout
    
    
    def output(self,data):
        # data: (B, N, dim)
        attn_input=layer_normalization(data) # (B,N,dim)
        attn_output,_=self.attn.output(attn_input,self.num_heads) # (B,N,dim)
        attn_output=tf.nn.dropout(attn_output,self.dropout) # (B,N,dim)
        residual_input=data+attn_output # (B,N,dim)
        mlp_input=layer_normalization(residual_input) # (B,N,dim)
        mlp_output=self.activation(self.mlp1.output(mlp_input)) # (B,N,dim)
        mlp_output=tf.nn.dropout(mlp_output,self.dropout) # (B,N,dim)
        mlp_output=self.mlp2.output(mlp_output)
        mlp_output=self.mlp_dropout(mlp_output,self.dropout)
        output=residual_input+mlp_output # (B,N,dim)
        return output
