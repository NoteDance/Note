import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.batch_norm import batch_norm
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.dropout import dropout
from Note.nn.layer.stochastic_depth import stochastic_depth
from Note.nn.layer.identity import identity
from Note.nn.initializer import initializer_
from Note.nn.activation import activation_dict
from Note.nn.Layers import Layers
from Note.nn.parallel.optimizer import AdamW
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Module import Module


class SwiftFormer:

    def __init__(self, model_type,
                 mlp_ratios=4, downsamples=[True, True, True, True],
                 act_layer=activation_dict['gelu'],
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=1,
                 distillation=True,
                 include_top=True,
                 pooling=None,
                 epsilon=1e-8,
                 weight_decay=0.025,
                 device='GPU',
                 dtype='float32'
                 ):
        Module.init()
        
        layers=SwiftFormer_depth[model_type]
        embed_dims=SwiftFormer_width[model_type]
        self.layers_dict={}

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.include_top=include_top
        self.pooling=pooling
        self.epsilon=epsilon
        self.weight_decay=weight_decay
        self.device=device
        self.alpha = tf.constant(0.5, dtype)
        self.temperature = tf.constant(10, dtype)
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.km=0

        self.patch_embed = stem(3, embed_dims[0], dtype)

        self.network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios,
                          act_layer=act_layer,
                          drop_rate=drop_rate,
                          drop_path_rate=drop_path_rate,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          vit_num=vit_num)
            self.network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                self.network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1], dtype=dtype
                    )
                )

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0:
                    layer = identity()
                else:
                    layer = batch_norm(embed_dims[i_emb],dtype=dtype)
                layer_name = f'norm{i_layer}'
                self.layers_dict[layer_name]=layer
        else:
            # Classifier head
            self.norm = batch_norm(embed_dims[-1],dtype=dtype)
            self.head = dense(
                num_classes, embed_dims[-1], weight_initializer=['truncated_normal',.02], dtype=dtype) if num_classes > 0 \
                else identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = dense(
                    num_classes, embed_dims[-1], weight_initializer=['truncated_normal',.02], dtype=dtype) if num_classes > 0 \
                    else identity()
        
        self.dtype=dtype
        self.optimizer=AdamW(epsilon=epsilon,weight_decay=weight_decay)
        self.param=Module.param
    
    
    def fine_tuning(self,classes=None,lr=None,flag=0):
        param=[]
        if flag==0:
            self.param_=self.param
            self.head_=self.head
            self.head=dense(
                classes, self.head.input_size, weight_initializer=['truncated_normal',.02], dtype=self.head.dtype) if classes > 0 \
                else identity()
            if self.dist:
                self.dist_head_=self.dist_head
                self.dist_head = dense(
                    classes, self.dist_head.input_size, weight_initializer=['truncated_normal',.02], dtype=self.dist_head.dtype) if classes > 0 \
                    else identity()
                param.extend(self.dist_head.param)
            param.extend(self.head.param)
            self.param=param
            self.optimizer_=self.optimizer
            self.optimizer=AdamW(lr=lr,param=self.param)
        elif flag==1:
            if self.dist:
                del self.param_[-len(self.dist_head.param):]
            del self.param_[-len(self.head.param):]
            self.param_.extend(self.head.param)
            if self.dist:
                self.param_.extend(self.dist_head.param)
            self.param=self.param_
        else:
            self.head,self.head_=self.head_,self.head
            if self.dist:
                self.dist_head,self.dist_head_=self.dist_head_,self.dist_head
                del self.param_[-len(self.dist_head.param):]
            del self.param_[-len(self.head.param):]
            self.param_.extend(self.head.param)
            if self.dist:
                self.param_.extend(self.dist_head.param)
            self.param=self.param_
        return


    def forward_tokens(self, x, train_flag=True):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x,train_flag)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = self.layers_dict[f'norm{idx}']
                if hasattr(norm_layer, 'train_flag'):
                    x_out = norm_layer(x,train_flag)
                else:
                    x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x
    

    def fp(self, x, p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x = self.patch_embed(x)
                x = self.forward_tokens(x)
                if self.fork_feat:
                    # Output features of four stages for dense prediction
                    return x
        
                x = self.norm(x)
                if self.include_top:
                    x = tf.transpose(x,perm=[0,3,1,2])
                    if self.dist:
                        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
                        x = tf.reduce_mean(x, axis=-1)
                        cls_out = self.head(x), self.dist_head(x)
                        if not self.km:
                            cls_out = (cls_out[0] + cls_out[1]) / 2
                    else:
                        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
                        x = tf.reduce_mean(x, axis=-1)
                        cls_out = self.head(x)
                    # For image classification
                    return cls_out
                else:
                    if self.pooling=="avg":
                        x = tf.math.reduce_mean(x, axis=[1, 2])
                    else:
                        x = tf.math.reduce_max(x, axis=[1, 2])
                    return x
        else:
            x = self.patch_embed(x,self.km)
            x = self.forward_tokens(x,self.km)
            if self.fork_feat:
                # Output features of four stages for dense prediction
                return x
    
            x = self.norm(x,self.km)
            if self.include_top:
                x = tf.transpose(x,perm=[0,3,1,2])
                if self.dist:
                    x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
                    x = tf.reduce_mean(x, axis=-1)
                    cls_out = self.head(x), self.dist_head(x)
                    if not self.km:
                        cls_out = (cls_out[0] + cls_out[1]) / 2
                else:
                    x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
                    x = tf.reduce_mean(x, axis=-1)
                    cls_out = self.head(x)
                # For image classification
                return cls_out
            else:
                if self.pooling=="avg":
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                else:
                    x = tf.math.reduce_max(x, axis=[1, 2])
                return x

    
    
    def distill_loss(self, student, teacher, temperature):
        student = tf.nn.softmax(student / temperature, axis=-1)
        teacher = tf.nn.softmax(teacher / temperature, axis=-1)
        return -tf.reduce_mean(tf.reduce_sum(teacher * tf.math.log(student + 1e-12), axis=-1))
    
    
    def total_loss(self, output, labels, alpha, temperature):
        if self.dist:
            cls_out, dist_out = output
            ce = self.ce_loss(labels, cls_out)
            distill = self.distill_loss(dist_out, labels, temperature)
            return alpha * ce + (1 - alpha) * distill
        else:
            ce = self.ce_loss(labels, output)
            return ce
    
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,self.device)):
            loss_value = self.total_loss(output, labels, self.alpha, self.temperature)
            return loss_value
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
            return tape,output,loss
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,self.device)):
            param=self.optimizer(gradient,self.param,self.bc[0])
            return param
    
    
SwiftFormer_width = {
    'XS': [48, 56, 112, 220],
    'S': [48, 64, 168, 224],
    'l1': [48, 96, 192, 384],
    'l3': [64, 128, 320, 512],
}


SwiftFormer_depth = {
    'XS': [3, 3, 6, 4],
    'S': [3, 3, 9, 6],
    'l1': [4, 3, 10, 5],
    'l3': [4, 4, 12, 6],
}


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, int):
        return (x, x)
    else:
        raise ValueError("Unsupported type for to_2tuple: {}".format(type(x)))


def stem(in_chs, out_chs, dtype):
    """
    Stem Layer that is implemented by two layers of conv.
    """
    layers=Layers()
    layers.add(conv2d(out_chs // 2, kernel_size=3, input_size=in_chs, strides=2, dtype=dtype))
    layers.add(zeropadding2d(padding=1))
    layers.add(batch_norm(dtype=dtype))
    layers.add(activation_dict['relu'])
    layers.add(conv2d(out_chs, kernel_size=3, strides=2, dtype=dtype))
    layers.add(zeropadding2d(padding=1))
    layers.add(batch_norm(dtype=dtype))
    layers.add(activation_dict['relu'])
    return layers


class Embedding:
    """
    Patch Embedding that is implemented by a layer of conv.
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=batch_norm, dtype='float32'):
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = conv2d(embed_dim, kernel_size=patch_size, input_size=in_chans,
                              strides=stride, dtype=dtype)
        self.zeropadding2d=zeropadding2d(padding=padding)
        self.norm = batch_norm(embed_dim, dtype=dtype) if norm_layer else identity()
        self.train_flag=True
    
    
    def __call__(self, x, train_flag=True):
        x = self.proj(x)
        if hasattr(self.norm, 'train_flag'):
            x = self.norm(x, train_flag)
        else:
            x = self.norm(x)
        return x


class ConvEncoder:
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    """

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True, dtype='float32'):
        self.layers=Layers()
        self.layers.add(depthwise_conv2d(kernel_size=kernel_size, input_size=dim, 
                                       weight_initializer=['truncated_normal',.02], dtype=dtype))
        self.layers.add(zeropadding2d(padding=kernel_size // 2))
        self.layers.add(batch_norm(dtype=dtype))
        self.layers.add(conv2d(hidden_dim, kernel_size=1, 
                              weight_initializer=['truncated_normal',.02], dtype=dtype))
        self.layers.add(activation_dict['gelu'])
        self.layers.add(conv2d(dim, kernel_size=1, 
                              weight_initializer=['truncated_normal',.02], dtype=dtype))
        self.drop_path = stochastic_depth(drop_path) if drop_path > 0. \
            else identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = initializer_([dim],'ones',dtype)
        self.train_flag=True


    def __call__(self, x, train_flag=True):
        input = x
        x = self.layers(x,train_flag)
        if self.use_layer_scale:
            if hasattr(self.drop_path, 'train_flag'):
                x = input + self.drop_path(self.layer_scale * x, train_flag)
            else:
                x = input + self.drop_path(self.layer_scale * x)
        else:
            if hasattr(self.drop_path, 'train_flag'):
                x = input + self.drop_path(x, train_flag)
            else:
                x = input + self.drop_path(x)
        return x


class Mlp:
    """
    Implementation of MLP layer with 1*1 convolutions.
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=activation_dict['gelu'], drop=0., dtype='float32'):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layers=Layers()
        self.layers.add(batch_norm(in_features, dtype=dtype))
        self.layers.add(conv2d(hidden_features, 1, weight_initializer=['truncated_normal',.02],
                                  dtype=dtype))
        self.layers.add(act_layer)
        self.layers.add(conv2d(out_features, 1, weight_initializer=['truncated_normal',.02],
                               dtype=dtype))
        self.layers.add(dropout(drop))


    def __call__(self, x, train_flag=True):
        x=self.layers(x,train_flag)
        return x


class EfficientAdditiveAttnetion:
    """
    Efficient Additive Attention module for SwiftFormer.
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2, dtype='float32'):
        self.to_query = dense(token_dim * num_heads, in_dims, dtype=dtype)
        self.to_key = dense(token_dim * num_heads, in_dims, dtype=dtype)

        self.w_g = initializer_([token_dim * num_heads, 1],'normal',dtype)
        self.scale_factor = token_dim ** -0.5
        self.Proj = dense(token_dim * num_heads, token_dim * num_heads)
        self.final = dense(token_dim, token_dim * num_heads)


    def __call__(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = tf.math.l2_normalize(query, axis=-1) #BxNxD
        key = tf.math.l2_normalize(key, axis=-1) #BxNxD

        query_weight = tf.matmul(query,self.w_g) # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = tf.math.l2_normalize(A, axis=1) # BxNx1

        G = tf.math.reduce_sum(A * query, axis=1) # BxD

        repeat = tf.shape(key)[1]
        multiples = tf.convert_to_tensor([1, repeat])
        G = tf.tile(G, multiples)
        G = tf.reshape(G, (tf.shape(G)[0], repeat, -1)) # BxNxD

        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out


class SwiftFormerLocalRepresentation:
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True, dtype='float32'):
        self.layers=Layers()
        self.layers.add(depthwise_conv2d(kernel_size=kernel_size, input_size=dim, weight_initializer=['truncated_normal',.02], dtype=dtype))
        self.layers.add(zeropadding2d(padding=kernel_size // 2))
        self.layers.add(batch_norm(dtype=dtype))
        self.layers.add(conv2d(dim, kernel_size=1, weight_initializer=['truncated_normal',.02], dtype=dtype))
        self.layers.add(activation_dict['gelu'])
        self.layers.add(conv2d(dim, kernel_size=1, weight_initializer=['truncated_normal',.02], dtype=dtype))
        self.drop_path = stochastic_depth(drop_path) if drop_path > 0. \
            else identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = initializer_([dim],'ones',dtype)


    def __call__(self, x, train_flag=True):
        input = x
        x = self.layers(x,train_flag)
        if self.use_layer_scale:
            if hasattr(self.drop_path, 'train_flag'):
                x = input + self.drop_path(self.layer_scale * x, train_flag)
            else:
                x = input + self.drop_path(self.layer_scale * x)
        else:
            if hasattr(self.drop_path, 'train_flag'):
                x = input + self.drop_path(x, train_flag)
            else:
                x = input + self.drop_path(x)
        return x


class SwiftFormerEncoder:
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=activation_dict['gelu'],
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, dtype='float32'):
        self.local_representation = SwiftFormerLocalRepresentation(dim=dim, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = stochastic_depth(drop_path) if drop_path > 0. \
            else identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = initializer_([dim],'ones',dtype)
            
            self.layer_scale_2 = initializer_([dim],'ones',dtype)
            
        self.train_flag=True
        

    def __call__(self, x, train_flag=True):
        x = self.local_representation(x,train_flag)
        B, H, W, C = x.shape
        if self.use_layer_scale:
            if hasattr(self.drop_path, 'train_flag'):
                x = tf.transpose(x, perm=[0, 2, 3, 1])
                x = tf.reshape(x, shape=(B, H * W, C))
                x=self.attn(x)
                x = tf.reshape(x, shape=(B, H, W, C))
                x = x + self.drop_path(self.layer_scale_1 * x, train_flag)
                x = x + self.drop_path(self.layer_scale_2 * self.linear(x, train_flag), train_flag)
            else:
                x = tf.transpose(x, perm=[0, 2, 3, 1])
                x = tf.reshape(x, shape=(B, H * W, C))
                x=self.attn(x)
                x = tf.reshape(x, shape=(B, H, W, C))
                x = x + self.drop_path(self.layer_scale_1 * x)
                x = x + self.drop_path(self.layer_scale_2 * self.linear(x, train_flag)) 
        else:
            if hasattr(self.drop_path, 'train_flag'):
                x = tf.transpose(x, perm=[0, 2, 3, 1])
                x = tf.reshape(x, shape=(B, H * W, C))
                x=self.attn(x)
                x = tf.reshape(x, shape=(B, H, W, C))
                x = x + self.drop_path(x, train_flag)
                x = x + self.drop_path(self.linear(x, train_flag), train_flag)
            else:
                x = tf.transpose(x, perm=[0, 2, 3, 1])
                x = tf.reshape(x, shape=(B, H * W, C))
                x=self.attn(x)
                x = tf.reshape(x, shape=(B, H, W, C))
                x = x + self.drop_path(x)
                x = x + self.drop_path(self.linear(x, train_flag))
        return x


def Stage(dim, index, layers, mlp_ratio=4.,
          act_layer=activation_dict['gelu'],
          drop_rate=.0, drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1, dtype='float32'):
    """
    Implementation of each SwiftFormer stages. Here, SwiftFormerEncoder used as the last block in all stages, while ConvEncoder used in the rest of the blocks.
    """
    blocks = Layers()

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        if layers[index] - block_idx <= vit_num:
            blocks.add(SwiftFormerEncoder(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,dtype=dtype))

        else:
            blocks.add(ConvEncoder(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3 ,dtype=dtype))

    return blocks