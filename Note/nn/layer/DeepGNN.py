class DeepGNN:
    def __init__(self, num_layers,layers,layer_params):
        # num_layers: the number of layers in the graph neural network
        # layer_types: the types of each layer in the graph neural network, such as "GCN", "GAT", etc.
        # layer_params: the parameters of each layer in the graph neural network, such as input and output dimensions, activation functions, etc.
        assert num_layers==len(layers)==len(layer_params)
        self.num_layers=num_layers
        self.layers=[]
        for i in range(num_layers):
            # create the corresponding graph neural network layer according to layer_types and layer_params
            layer=layers[i]
            layer_param=layer_params[i]
            self.layers.append(layer(**layer_param))
    
    
    def output(self,graph,data):
        # graph: a dictionary that represents the input graph structure data, containing adjacency matrix, node features, etc.
        # data: a tensor that represents the input node features
        output=data
        for i in range(self.num_layers):
            # execute the graph neural network operation for each layer
            layer=self.layers[i]
            output=layer(graph,output)
        return output