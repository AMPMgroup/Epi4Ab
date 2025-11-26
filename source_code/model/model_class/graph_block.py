from torch_geometric.nn import ChebConv, GATConv, GCNConv, Sequential
from torch_geometric.nn.norm import BatchNorm, GraphNorm
from torch.nn import Module, LeakyReLU, Dropout, Linear, LayerNorm

class GNNNaiveBlock_Cheb(Module):
    def __init__(self, in_hidden_channel, out_hidden_channel, drop_out, filter_size,
                norm):
        super(GNNNaiveBlock_Cheb, self).__init__()
        self.graph = ChebConv(in_hidden_channel,
                            out_hidden_channel,
                            K = filter_size)
        self.act = LeakyReLU()
        self.dropout = Dropout(drop_out)
        self.norm = norm

    def forward(self, x, edgeIndex, edgeAttribute):
        if self.norm:
            x = self.norm(x)
        x_out = self.dropout(self.act(self.graph(x, edgeIndex, edgeAttribute)))

        return x_out

class GNNNaiveBlock_ChebGat(Module):
    def __init__(self, in_hidden_channel, out_hidden_channel, drop_out, filter_size, attention_head,
                 norm):
        super(GNNNaiveBlock_ChebGat, self).__init__()
        self.graph = ChebConv(in_hidden_channel,
                            out_hidden_channel,
                            K = filter_size)
        self.act1 = LeakyReLU()
        self.dropout1 = Dropout(drop_out)
        self.att_graph = GATConv(out_hidden_channel, 
                                out_hidden_channel, 
                                heads = attention_head, 
                                concat = False)
        self.act2 = LeakyReLU()
        self.dropout2 = Dropout(drop_out)
        self.norm = norm

    def forward(self, x, edgeIndex, edgeAttribute):
        if self.norm:
            x = self.norm(x)
        x_out = self.dropout1(self.act1(self.graph(x, edgeIndex, edgeAttribute)))
        x_out = self.dropout2(self.act2(self.att_graph(x_out, edgeIndex, edgeAttribute)))

        return x_out
    
class GNNNaiveBlock_Base(Module):
    def __init__(self, in_hidden_channel, out_hidden_channel, drop_out,
                 norm):
        super(GNNNaiveBlock_Base, self).__init__()
        self.graph = GCNConv(in_hidden_channel,
                            out_hidden_channel)
        self.act = LeakyReLU()
        self.dropout = Dropout(drop_out)
        self.norm = norm

    def forward(self, x, edgeIndex, edgeAttribute):
        if self.norm:
            x = self.norm(x)
        x_out = self.dropout(self.act(self.graph(x, edgeIndex, edgeAttribute)))

        return x_out

class GNNResNetBlock_Base(Module):
    def __init__(self, in_hidden_channel, out_hidden_channel, drop_out,
                 norm):
        super(GNNResNetBlock_Base, self).__init__()
        self.graph = GCNConv(in_hidden_channel,
                            out_hidden_channel)
        self.act = LeakyReLU()
        self.dropout = Dropout(drop_out)
        self.linear = Linear(in_hidden_channel, out_hidden_channel)
        self.norm = norm
        # self.feed_forward = Sequential('x, edgeIndex, edgeAttribute', [
        #     (GCNConv(in_hidden_channel,
        #               out_hidden_channel), 'x, edgeIndex, edgeAttribute -> x1'),
        #     (LeakyReLU(), 'x1 -> x1a'),
        #     (Dropout(drop_out), 'x1a -> x1d'),
        #     (Linear(in_hidden_channel, out_hidden_channel), 'x -> x'),
        #     (lambda x1, x2: x1 + x2, 'x, x1d -> xs'),
        #     (LayerNorm(out_hidden_channel), 'xs -> xsn')
        # ])

    def forward(self, x, edgeIndex, edgeAttribute):
        if self.norm:
            x = self.norm(x)
        x_out = self.dropout(self.act(self.graph(x, edgeIndex, edgeAttribute)))
        x_trans = self.linear(x)
        x_out = x_out + x_trans
        return x_out

class GNNResNetBlock_Cheb(Module):
    def __init__(self, in_hidden_channel, out_hidden_channel, drop_out, filter_size,
                 norm):
        super(GNNResNetBlock_Cheb, self).__init__()
        self.graph = ChebConv(in_hidden_channel,
                            out_hidden_channel,
                            K = filter_size)
        self.act = LeakyReLU()
        self.dropout = Dropout(drop_out)
        self.linear = Linear(in_hidden_channel, out_hidden_channel)
        self.norm = norm

    def forward(self, x, edgeIndex, edgeAttribute):
        if self.norm:
            x = self.norm(x)
        x_graph = self.graph(x, edgeIndex, edgeAttribute)
        x_graph = self.dropout(self.act(x_graph))
        x = self.linear(x)
        x_out = x + x_graph
        # x_out = self.feed_forward(x, edgeIndex, edgeAttribute)
        return x_out

class GNNResNetBlock_ChebGat(Module):
    def __init__(self, in_hidden_channel, out_hidden_channel, drop_out, filter_size, attention_head, gat_concat,
                 norm):
        '''
        Cheb - GAT - initial adding
        '''
        super(GNNResNetBlock_ChebGat, self).__init__()
        if gat_concat:
            assert out_hidden_channel % attention_head == 0, f'{out_hidden_channel} should be divisible by {attention_head}.'
            gat_out_channel = int(out_hidden_channel / attention_head)
        else:
            gat_out_channel = out_hidden_channel
        self.cheb_conv = ChebConv(in_hidden_channel, out_hidden_channel, K = filter_size)
        self.gat_conv = GATConv(out_hidden_channel, gat_out_channel, heads = attention_head, concat = gat_concat)
        self.linear = Linear(in_hidden_channel, out_hidden_channel)
        self.activation1 = LeakyReLU()
        self.activation2 = LeakyReLU()
        self.dropout1 = Dropout(drop_out)
        self.dropout2 = Dropout(drop_out)
        self.norm = norm

    def forward(self, x, edgeIndex, edgeAttribute):
        # x_out = self.feed_forward(x, edgeIndex, edgeAttribute)
        if self.norm:
            x = self.norm(x)
        x_out = self.dropout1(self.activation1(self.cheb_conv(x, edgeIndex, edgeAttribute)))
        x_out = self.dropout2(self.activation2(self.gat_conv(x_out, edgeIndex, edgeAttribute)))
        x = self.linear(x)
        x_out = x + x_out

        return x_out