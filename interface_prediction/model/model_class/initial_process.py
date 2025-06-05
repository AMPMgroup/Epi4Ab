import torch
from torch import nn

class InitialProcess(nn.Module):
    def __init__(self,
                 use_pretrained,
                 pretrained_model,
                 freeze_pretrained,
                 use_seq_ff,
                 seq_ff_in,
                 seq_ff_dim,
                 seq_ff_out,
                 seq_ff_dropout,
                 use_antiberty,
                 antiberty_ff_in,
                 antiberty_ff_dim,
                 antiberty_ff_out,
                 antiberty_ff_dropout,
                 antiberty_max_len,
                 use_token,
                 vh_token_size,
                 vl_token_size,
                 token_dim,
                 use_struct,
                 use_deep_shallow,
                 shallow_cutoff,
                 resDepth_index,
                 initial_process_weight_dict,
                 use_mha,
                 mha_head,
                 max_antigen_len,
                 in_feature,
                 device):
        super(InitialProcess, self).__init__()
        self.use_pretrained = use_pretrained
        self.use_seq_ff = use_seq_ff
        if self.use_pretrained:
            if use_seq_ff:
                self.seq_linear1 = nn.Linear(seq_ff_in, seq_ff_dim)
                self.seq_dropout = nn.Dropout(seq_ff_dropout)
                self.seq_activation = nn.ReLU()
                self.seq_linear2 = nn.Linear(seq_ff_dim, seq_ff_out)
        self.freeze_pretrained = freeze_pretrained
        if self.use_pretrained & (not self.freeze_pretrained):
            if pretrained_model == 'protBERT':
                from transformers import BertModel, BertTokenizer
                self.pretrained_tokenize = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
                self.pretrained_model = BertModel.from_pretrained("Rostlab/prot_bert")
            elif pretrained_model[:4] == 'ESM2':
                from transformers import AutoTokenizer, EsmModel
                if pretrained_model == 'ESM2_t6':
                    self.pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
                    self.pretrained_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                elif pretrained_model == 'ESM2_t12':
                    self.pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
                    self.pretrained_model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
                elif pretrained_model == 'ESM2_t30':
                    self.pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
                    self.pretrained_model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
                elif pretrained_model == 'ESM2_t33':
                    self.pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
                    self.pretrained_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
                elif pretrained_model == 'ESM2_t36':
                    self.pretrained_tokenize = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
                    self.pretrained_model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.use_antiberty = use_antiberty
        self.antiberty_max_len = antiberty_max_len
        if self.use_antiberty:
            self.linear1 = nn.Linear(antiberty_ff_in, antiberty_ff_dim)
            self.dropout = nn.Dropout(antiberty_ff_dropout)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(antiberty_ff_dim, antiberty_ff_out)
        self.use_token = use_token
        self.use_struct = use_struct
        self.use_deep_shallow = use_deep_shallow
        self.shallow_cutoff = shallow_cutoff
        self.resDepth_index = resDepth_index
        self.initial_process_weight_dict = initial_process_weight_dict
        if self.use_token:
            self.vh_token_embed = nn.Embedding(vh_token_size, token_dim)
            self.vl_token_embed = nn.Embedding(vl_token_size, token_dim)
        self.use_mha = use_mha
        if self.use_mha:
            self.mha = nn.MultiheadAttention(in_feature, mha_head, 
                                        vdim=antiberty_ff_in, kdim=antiberty_ff_in)
        self.max_antigen_len = max_antigen_len
        self.in_feature = in_feature
        self.device = device

    def run_pretrained(self, seq):
        encode_input = self.pretrained_tokenize(seq, return_tensors = 'pt')
        encode_input = {k:v.to(self.device) for k,v in encode_input.items()}
        output = self.pretrained_model(**encode_input)
        output = output[0].reshape(output[0].shape[1],-1)[1:-1,:]
        return output

    def forward(self, x_struct, x_seq, x_antiberty, token_seq, node_size):
        if self.use_pretrained:
            if not self.freeze_pretrained:
                if isinstance(x_seq, list):
                    temp_seq = []
                    for seq in x_seq:
                        temp_seq.append(self.run_pretrained(seq))
                    x_seq = torch.cat(temp_seq, dim = 0)
                else:
                    x_seq = self.run_pretrained(x_seq)
            if self.use_seq_ff:
                x_seq = self.seq_linear2(self.seq_dropout(self.seq_activation(self.seq_linear1(x_seq))))
            x_seq = x_seq * self.initial_process_weight_dict['pre-trained']
            seq_len = x_seq.size(0)
        else:
            x_seq = None

        if self.use_struct:
            if self.use_deep_shallow:
                shallow_index = torch.where(x_struct[:,self.resDepth_index]<=self.shallow_cutoff)[0]
            else:
                shallow_index = None
            x_struct = x_struct * self.initial_process_weight_dict['struct']
            seq_len = x_struct.size(0)
        else:
            x_struct = None
            shallow_index = None
            deep_index = None

        if self.use_token:
            token_feature = []
            for token, pdb_size in zip(token_seq, node_size):
                vh_token_feature = self.vh_token_embed(token[0])
                vl_token_feature = self.vl_token_embed(token[1])
                token_feature.append(torch.cat([vh_token_feature, vl_token_feature]).expand(pdb_size, -1))
            token_feature = torch.cat(token_feature)* self.initial_process_weight_dict['token']
        else:
            token_feature = None
        
        if self.use_mha:
            x_list = [i for i in [x_struct, x_seq, token_feature] if i is not None]
            x = torch.cat(x_list, dim=1) if len(x_list) > 1 else x_list[0]
            if len(node_size) != 1:
                x = torch.stack([torch.cat((ps, torch.zeros(self.max_antigen_len - ns, self.in_feature).to(self.device)),0) 
                            for ps, ns in zip(torch.split(x, node_size.tolist()), node_size)])
                x = x.permute(1,0,2)
                x_antiberty = torch.stack(torch.split(x_antiberty, self.antiberty_max_len))
                x_antiberty = x_antiberty.permute(1,0,2)
                x = self.mha(x, x_antiberty, x_antiberty)[0].permute(1,0,2)
                x = torch.cat([xs[:ns,:] for xs, ns in zip(x, node_size.tolist())])
            else:
                x = self.mha(x, x_antiberty, x_antiberty)[0]
            
        else:
            if self.use_antiberty:
                x_antiberty = torch.stack(torch.split(x_antiberty, [self.antiberty_max_len]*len(node_size)))
                x_antiberty = self.linear2(self.dropout(self.activation(self.linear1(x_antiberty)))).flatten(1)
                x_antiberty = torch.cat([i.expand(s, -1) for i,s in zip(x_antiberty,node_size)])* self.initial_process_weight_dict['antiberty']
            else:
                x_antiberty = None

            assert (x_seq is not None) | (x_struct is not None) | (token_feature is not None), f'{x_seq}, {x_struct} and {token_feature} cannot be all None.'

            x_list = [i for i in [x_struct, x_seq, x_antiberty, token_feature] if i is not None]
            x = torch.cat(x_list, dim=1) if len(x_list) > 1 else x_list[0]
        
        return x, shallow_index

