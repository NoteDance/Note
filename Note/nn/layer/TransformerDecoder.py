class TransformerDecoder:
    def __init__(self, decoder_layers, num_layers, norm=None):
        self.layers = decoder_layers
        self.num_layers = num_layers
        self.norm = norm


    def __call__(self, tgt, memory, tgt_mask = None,
                memory_mask = None, train_flag=True):
        output = tgt

        for mod in self.layers:
            output = mod.output(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask, train_flag=train_flag
                         )

        if self.norm is not None:
            output = self.norm.output(output)

        return output