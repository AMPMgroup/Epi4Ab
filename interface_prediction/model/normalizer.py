from torch_geometric.nn.norm import BatchNorm, GraphNorm

class Normalizer:
    def __init__(self, 
                 block_norm,
                 block_norm_eps,
                 block_norm_momentum):
        self.block_norm = block_norm
        self.block_norm_eps = block_norm_eps
        self.block_norm_momentum = block_norm_momentum
        self.use_norm = True if self.block_norm else False

    def __call__(self, hidden_channel):
        if self.block_norm == 'batchnorm':
            return BatchNorm(hidden_channel, self.block_norm_eps, self.block_norm_momentum)
        elif self.block_norm == 'graphnorm':
            return GraphNorm(hidden_channel, self.block_norm_eps)
        else:
            return None