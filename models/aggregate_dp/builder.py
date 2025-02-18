import torch.nn as nn
from .gem import GeM_mask
from .netvlad import NetVLAD_mask
from .pos_gem import PoS_GeM
from .boq import BoQ
import torch


class aggregator(nn.Module):

    def __init__(self, aggregate_type, aggregate_cfgs, out_dim):
        super(aggregator, self).__init__()
        if aggregate_type == 'GeM':
            self.module = GeM_mask(aggregate_cfgs)
        elif aggregate_type == 'NetVLAD':
            self.module = NetVLAD_mask(aggregate_cfgs, out_dim)
        elif aggregate_type == 'PoS_GeM':
            self.module = PoS_GeM(aggregate_cfgs)
        elif aggregate_type == 'BoQ':
            self.module = BoQ(aggregate_cfgs)
        else:
            raise ValueError('Unknown aggregation type: {}'.format(aggregate_type))
        self.aggregate_type = aggregate_type
    
    def forward(self, x, mask=None, index_list=None, coords_list=None):
        if mask is None and self.aggregate_type != 'PoS_GeM':
            mask = torch.ones_like(x[:, 0, ...], dtype=torch.bool)
        if self.aggregate_type == 'PoS_GeM':
            return self.module(x, index_list, coords_list)
        elif self.aggregate_type == 'BoQ':
            output, _ = self.module(x)
            return output
        else:
            return self.module(x, mask)