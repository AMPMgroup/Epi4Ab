import torch

class AATokenizer:
    def __init__(self, feature_dict, aa_list=None, vh_list=None, vl_list=None, para_token_list=None, para_two_sequences=False):
        assert (len(feature_dict) != 0) | ((aa_list is not None) & (vh_list is not None) & (vl_list is not None)), f'{feature_dict}, {aa_list}, {vh_list} and {vl_list} can not be all None.'
        if len(feature_dict) == 0:
            self.aa_list = aa_list
            self.vh_list = vh_list
            self.vl_list = vl_list
        else:
            self.aa_list = feature_dict['AA_short']
            self.vh_list = feature_dict['Ab_features']['onehot_vh']
            self.vl_list = feature_dict['Ab_features']['onehot_vl']
        self.aa_len = len(self.aa_list)
        self.vh_len = len(self.vh_list)
        self.vl_len = len(self.vl_list)
        self.para_token_list = para_token_list
        self.para_two_sequences = para_two_sequences

    def tokenize_feature(self, aa_src, vh_fam, vl_fam):
        aa_index = [self.aa_list.index(aa) for aa in aa_src]
        vh_index = self.vh_list.index(vh_fam)
        vl_index = self.vl_list.index(vl_fam)
        ah_token = torch.tensor([vh_index*self.vl_len*self.aa_len + self.aa_len*vl_index + aa for aa in aa_index])
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