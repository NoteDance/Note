class TransformerEncoder:
    def __init__(self, encoder_layer, num_layers, norm=None):
        self.layers = [encoder_layer for _ in range(num_layers)]
        self.num_layers = num_layers
        self.norm = norm


    def output(
            self,
            src,
            mask = None,
            ):
        output = src

        for mod in self.layers:
            output = mod.output(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm.output(output)

        return output