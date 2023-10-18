class TransformerDecoder:
    def __init__(self, decoder_layer, num_layers, norm=None):
        self.layers = [decoder_layer for _ in range(num_layers)]
        self.num_layers = num_layers
        self.norm = norm


    def output(self, tgt, memory, tgt_mask = None,
                memory_mask = None):
        output = tgt

        for mod in self.layers:
            output = mod.output(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         )

        if self.norm is not None:
            output = self.norm.output(output)

        return output