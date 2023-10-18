class TransformerEncoder:
    def __init__(self, encoder_layers, num_layers, norm=None):
        self.layers = encoder_layers
        self.num_layers = num_layers
        self.norm = norm


    def output(
            self,
            src,
            mask = None,
            train_flag=True
            ):
        output = src

        for mod in self.layers:
            output = mod.output(output, src_mask=mask, train_flag=train_flag)

        if self.norm is not None:
            output = self.norm.output(output)

        return output
