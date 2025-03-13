import torch
from torch import nn

class InitialProcess(nn.Module):
    def __init__(self,
                 combine_input,
                 use_pretrained,
                 pretrained_model,
                 freeze_pretrained,
                 use_token,
                 token_size,
                 token_dim,
                 use_continuous,
                 reserved_columns,
                 continuous_embed_dim,
                 use_struct,
                 initial_process_weight_dict,
                 device):
        super(InitialProcess, self).__init__()
        self.combine_input = combine_input
        self.use_pretrained = use_pretrained
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
        self.use_continuous = use_continuous
        self.use_token = use_token
        self.use_struct = use_struct
        self.initial_process_weight_dict = initial_process_weight_dict
        if self.use_continuous == 'embeded':
            self.struct_embed = nn.Linear(reserved_columns, continuous_embed_dim)
        if self.use_token:
            self.token_embed = nn.Embedding(token_size, token_dim)
        self.device = device

    def run_pretrained(self, seq):
        encode_input = self.pretrained_tokenize(seq, return_tensors = 'pt')
        encode_input = {k:v.to(self.device) for k,v in encode_input.items()}
        output = self.pretrained_model(**encode_input)
        output = output[0].reshape(output[0].shape[1],-1)[1:-1,:]
        return output

    def forward(self, x_struct, x_seq, token_seq):
        if self.use_pretrained:
            if not self.freeze_pretrained:
                if isinstance(x_seq, list):
                    temp_seq = []
                    for seq in x_seq:
                        temp_seq.append(self.run_pretrained(seq))
                    x_seq = torch.cat(temp_seq, dim = 0)
                else:
                    x_seq = self.run_pretrained(x_seq)
            x_seq = x_seq * self.initial_process_weight_dict['pre-trained']
        else:
            x_seq = None

        if self.use_continuous == 'embeded':
            x_struct = self.struct_embed(x_struct) * self.initial_process_weight_dict['struct']
        elif (self.use_continuous == 'absolute') | (self.use_struct):
            x_struct = x_struct * self.initial_process_weight_dict['struct']
        else:
            x_struct = None
        
        if self.use_token:
            token_feature = self.token_embed(token_seq) * self.initial_process_weight_dict['token']
        else:
            token_feature = None

        assert (x_seq is not None) | (x_struct is not None) | (token_feature is not None), f'{x_seq}, {x_struct} and {token_feature} cannot be all None.'

        x_list = [i for i in [x_struct, x_seq, token_feature] if i is not None]

        if self.combine_input == 'concat':
            x = torch.cat(x_list, dim=1) if len(x_list) > 1 else x_list[0]
        else:
            # NOTE: should consider divide by sum of weight
            x = torch.stack(x_list).sum(dim=0)/len(x_list) if len(x_list) > 1 else x_list[0]
        return x

