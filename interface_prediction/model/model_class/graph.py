from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dropout_edge
from torch import nn
import torch
from .graph_block import GNNNaiveBlock_Cheb, GNNNaiveBlock_ChebGat,GNNNaiveBlock_Base,\
    GNNResNetBlock_Cheb,GNNResNetBlock_ChebGat, GNNResNetBlock_Base

class GNNNaive(nn.Module):
    def __init__(self, 
                 initial_process,
                 in_feature, 
                 out_label, 
                 hidden_channel:list,
                 num_layers, 
                 drop_out:list, 
                 attention_head:list, 
                 filter_size:list,
                 gradient_attribute,
                 gradient_attribute_with_bias,
                 model_block:str,
                 use_base_model:bool,
                 normalizer,
                 ):
        '''
        Parameters:
            in_feature: input dimension
            out_label: output dimension
            num_layers: number of layers
            hidden_channel: dimension of hidden layers (List)
            attention_head: attention head of GAT
        '''
        super(GNNNaive, self).__init__()
        self.initial_process = initial_process
        self.gradient_attribute = gradient_attribute
        self.out_label = out_label
        layers_list = []
        num_layers = num_layers + 1
        channel_list = [in_feature] + hidden_channel
        # self.token_embed = nn.Embedding(token_size, token_dim)
        # self.seq_embed = GCNConv(1024, pretrained_feature_number)
        # self.embed_pretrained = embed_pretrained

        for i in range(1, num_layers):
            if len(hidden_channel) == 1:
                hidden_channel_ind = 1
            else:
                hidden_channel_ind = i
            if len(drop_out) == 1:
                drop_out_ind = 0
            else:
                drop_out_ind = i - 1
            if len(attention_head) == 1:
                attention_head_ind = 0
            else:
                attention_head_ind = i - 1
            if len(filter_size) == 1:
                filter_size_ind = 0
            else:
                filter_size_ind = i - 1
            
            if use_base_model:
                block = GNNNaiveBlock_Base(channel_list[hidden_channel_ind - 1],
                                            channel_list[hidden_channel_ind],
                                            drop_out[drop_out_ind],
                                            normalizer(channel_list[hidden_channel_ind - 1]))
            elif model_block == 'GNNNaiveBlock_Cheb':
                block = GNNNaiveBlock_Cheb(channel_list[hidden_channel_ind - 1],
                                            channel_list[hidden_channel_ind],
                                            drop_out[drop_out_ind],
                                            filter_size[filter_size_ind],
                                            normalizer(channel_list[hidden_channel_ind - 1]))
            elif model_block == 'GNNNaiveBlock_ChebGat':
                block = GNNNaiveBlock_ChebGat(channel_list[hidden_channel_ind - 1],
                                            channel_list[hidden_channel_ind],
                                            drop_out[drop_out_ind],
                                            filter_size[filter_size_ind],
                                            attention_head[attention_head_ind],
                                            normalizer(channel_list[hidden_channel_ind - 1]))
            else:
                # block = GNNBlock_03(channel_list[hidden_channel_ind - 1],
                #                     channel_list[hidden_channel_ind],
                #                     drop_out[drop_out_ind])
                raise ValueError(f'{model_block} is not supported for GNNMultiLayers model.')
                
            layers_list.append(block)
        self.layers = nn.ModuleList(layers_list)
        self.attribute_layer = nn.Linear(3,1, bias=gradient_attribute_with_bias)
        if use_base_model:
            self.fc_out = GCNConv(channel_list[hidden_channel_ind], 
                                    out_label)
        else:
            self.fc_out = GATConv(channel_list[hidden_channel_ind], 
                                    out_label, 
                                    heads = attention_head[attention_head_ind], # Take the last attention head in the list
                                    concat = False)
        self.use_norm = normalizer.use_norm
        if self.use_norm:
            self.norm = normalizer(channel_list[hidden_channel_ind])

    def forward(self, x_struct, x_seq, edgeIndex, edgeAttribute, x_antiberty, token_seq, node_size):
        if self.gradient_attribute:
            atb = torch.flatten(self.attribute_layer(edgeAttribute)).clamp(0)
            # atb = torch.where(atb < 0, 0, atb)
        else:
            atb = edgeAttribute
        assert not atb.isnan().any(), f'There is NaN value after attribute layer {atb}'

        x = self.initial_process(x_struct, x_seq, x_antiberty, token_seq, node_size)

        for layer in self.layers:
            x = layer(x, edgeIndex, atb)
            assert not x.isnan().any(), f'There is NaN value after sequential layer {x}'
        if self.use_norm:
            x = self.norm(x)
        out = self.fc_out(x, edgeIndex, atb)
        assert not out.isnan().any(), f'There is NaN value after fc_out {out}'
        return out

class GNNResNet(nn.Module):
    def __init__(self, 
                 initial_process, 
                 in_feature,
                 out_label, 
                 hidden_channel:list, 
                 num_layers, 
                 drop_out:list, 
                 dropout_edge_p,
                 attention_head:list, 
                 filter_size:list,
                 gradient_attribute,
                 gradient_attribute_with_bias,
                 model_block:str,
                 use_base_model:bool,
                 gat_concat:bool,
                 normalizer):
        '''
        Parameters:
            in_feature: input dimension
            out_label: output dimension
            num_layers: number of layers
            hidden_channel: dimension of hidden layers (List)
            attention_head: attention head of GAT
        '''
        super(GNNResNet, self).__init__()
        self.gradient_attribute = gradient_attribute
        self.out_label = out_label
        self.initial_process = initial_process
        self.dropout_edge_p = dropout_edge_p
        layers_list = []
        num_layers = num_layers + 1
        channel_list = [in_feature] + hidden_channel

        for i in range(1, num_layers):
            if len(hidden_channel) == 1:
                hidden_channel_ind = 1
            else:
                hidden_channel_ind = i
            if len(drop_out) == 1:
                drop_out_ind = 0
            else:
                drop_out_ind = i - 1
            if len(attention_head) == 1:
                attention_head_ind = 0
            else:
                attention_head_ind = i - 1
            if len(filter_size) == 1:
                filter_size_ind = 0
            else:
                filter_size_ind = i - 1
            
            if use_base_model:
                block = GNNResNetBlock_Base(channel_list[hidden_channel_ind - 1],
                                            channel_list[hidden_channel_ind],
                                            drop_out[drop_out_ind],
                                            normalizer(channel_list[hidden_channel_ind - 1]))
            elif model_block == 'GNNResNetBlock_Cheb':
                block = GNNResNetBlock_Cheb(channel_list[hidden_channel_ind - 1],
                                            channel_list[hidden_channel_ind],
                                            drop_out[drop_out_ind],
                                            filter_size[filter_size_ind],
                                            normalizer(channel_list[hidden_channel_ind - 1]))
            elif model_block == 'GNNResNetBlock_ChebGat':
                block = GNNResNetBlock_ChebGat(channel_list[hidden_channel_ind - 1],
                                                channel_list[hidden_channel_ind],
                                                drop_out[drop_out_ind],
                                                filter_size[filter_size_ind],
                                                attention_head[attention_head_ind],
                                                gat_concat,
                                                normalizer(channel_list[hidden_channel_ind - 1]))
            else:
                raise ValueError(f'{model_block} is not supported for GNNResNet model.')
                
            layers_list.append(block)
        self.layers = nn.ModuleList(layers_list)
        self.attribute_layer = nn.Linear(3,1, bias=gradient_attribute_with_bias)
        self.fc_out = nn.Linear(channel_list[hidden_channel_ind], 
                                out_label)
        self.use_norm = normalizer.use_norm
        if self.use_norm:
            self.norm = normalizer(channel_list[hidden_channel_ind])
    def forward(self, x_struct, x_seq, edgeIndex, edgeAttribute, x_antiberty, token_seq, node_size):
        if self.gradient_attribute:
            atb = torch.flatten(self.attribute_layer(edgeAttribute)).clamp(0)
        else:
            atb = edgeAttribute
        assert not atb.isnan().any(), f'There is NaN value after attribute layer {atb}'

        x = self.initial_process(x_struct, x_seq, x_antiberty, token_seq, node_size)

        for layer in self.layers:
            if self.dropout_edge_p is not None:
                edge_index, edge_mask = dropout_edge(edgeIndex, force_undirected=True, p=self.dropout_edge_p)
                edge_attribute = atb[edge_mask]
                x = layer(x, edge_index, edge_attribute)
            else:
                x = layer(x, edgeIndex, atb)
            assert not x.isnan().any(), f'There is NaN value after sequential layer {x}'
        if self.use_norm:
            x = self.norm(x)
        out = self.fc_out(x)
        assert not out.isnan().any(), f'There is NaN value after fc_out {out}'
        return out