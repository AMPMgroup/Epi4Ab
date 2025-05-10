import torch

class AATokenizer:
    def __init__(self, feature_dict, vh_list=None, vl_list=None):
        assert (len(feature_dict) != 0) | ((vh_list is not None) & (vl_list is not None)), f'{feature_dict}, {vh_list} and {vl_list} can not be all None.'
        if len(feature_dict) == 0:
            self.vh_list = vh_list
            self.vl_list = vl_list
        else:
            self.vh_list = feature_dict['Ab_features']['onehot_vh']
            self.vl_list = feature_dict['Ab_features']['onehot_vl']
        self.vh_len = len(self.vh_list)
        self.vl_len = len(self.vl_list)

    def tokenize_feature(self, vh_fam, vl_fam):
        vh_index = self.vh_list.index(vh_fam)
        vl_index = self.vl_list.index(vl_fam)
        ah_token = torch.tensor([vh_index, vl_index])
        return ah_token
    
    def str_to_token(self, aa_seq):
        return torch.tensor([self.para_token_list.index(aa) for aa in aa_seq.split(' ')])
    
    def tokenize_sequence(self, aa_seq):
        if self.para_two_sequences:
            aa_list = aa_seq.split(' SEP')[:-1]
            heavy_seq = ''.join(aa_list[:3])
            light_seq = 'SOS' + ''.join(aa_list[3:])
            heavy_token = self.str_to_token(heavy_seq)
            light_token = self.str_to_token(light_seq)
            return [heavy_token, light_token]
        else:
            # Remove SEP for full sequence
            aa_seq = aa_seq.replace(' SEP','')
            return self.str_to_token(aa_seq)