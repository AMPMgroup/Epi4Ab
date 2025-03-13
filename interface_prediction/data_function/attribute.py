import torch

class CalculateAttribute:
    def __init__(self,
                 device,
                 edge_type,
                 gradient_attribute,
                 attribute_no_bond,
                 attribute_no_lj,
                 attribute_no_charge,
                 attribute_weight,
                 bias_distance):
        self.device = device
        self.edge_type = edge_type
        self.gradient_attribute = gradient_attribute
        self.attribute_no_bond = attribute_no_bond
        self.attribute_no_lj = attribute_no_lj
        self.attribute_no_charge = attribute_no_charge
        self.attribute_weight = attribute_weight
        self.bias_distance = bias_distance
        self.sigma = 2 * 1.7 # VdW radius of C alpha
        self.d_equilibrium = self.sigma * 2**(1/6)

    def lennard_jones_potential(self, distance):
        sigma_over_distance = self.sigma / distance
        v = torch.pow(sigma_over_distance, torch.tensor(12)) - torch.pow(sigma_over_distance, torch.tensor(6))
        return v

    def conditional_lennard_jones(self, distance):
        v_min = self.lennard_jones_potential(self.d_equilibrium)
        v = self.lennard_jones_potential(distance)
        v = torch.where(distance < self.d_equilibrium, v - 2*v_min, -v)
        return v

    def expand_zero(att_length):
        return torch.zeros(att_length)
    
    def __call__(self, edge_attribute, edge_charge):
        if self.edge_type == 'dist':
            att_length = edge_attribute.size()[0]
            if self.attribute_no_bond: # Bond potential
                attribute_1 = self.expand_zero(att_length)
            else:
                denominator = edge_attribute - self.bias_distance 
                assert torch.gt(denominator, 0).all(), 'Edge Attribute - Bias_distance should be greater than 0.'
                attribute_1 = torch.reciprocal(denominator)
                del denominator

            if self.attribute_no_lj: # Lennard-Jones potential
                attribute_2 = self.expand_zero(att_length)
            else:
                attribute_2 = self.conditional_lennard_jones(edge_attribute)
                assert torch.ge(attribute_2, 0).all(), 'Lennard-Jones potential should be equal or greater than 0.'

            if self.attribute_no_charge:
                attribute_3 = self.expand_zero(att_length)
            else:
                attribute_3 = (torch.abs(edge_charge)/edge_attribute)
                assert torch.ge(attribute_3, 0).all(), 'Charge potential should be equal or greater than 0.'

            if self.gradient_attribute:
                attribute = torch.stack((attribute_1, attribute_2, attribute_3)).T
            else:
                attribute = attribute_1 * self.attribute_weight[0] + attribute_2 * self.attribute_weight[1] + attribute_3 * self.attribute_weight[2]
                attribute = torch.where(attribute < 0, 0.0, attribute)
            del attribute_1
            del attribute_2
            del attribute_3
        else:
            attribute = edge_attribute
        assert torch.is_tensor(attribute), 'Total attribute has to be tensor.'
        assert not attribute.isnan().any(), f'edge attribute has Nan {attribute}'
        assert torch.ge(attribute, 0).all(), f'edge attribute has negative value {attribute}'
        return attribute