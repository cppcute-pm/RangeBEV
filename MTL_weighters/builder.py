from .db_mtl import DB_MTL_v1
from .aligned_mtl import Aligned_MTL_v1
from .moco_mtl import MoCo_MTL_v1
from .stch_mtl import STCH_MTL_v1
from torch import nn
import torch


class None_MTL(nn.Module):
    def __init__(self, cfgs, model_loss, device):
        super(None_MTL, self).__init__()
    
    def load_state_dict(self, state_dict):
        pass

    def get_state_dict(self,):
        state_dict = {}
        return state_dict

    def backward(self, losses, model_loss, epoch=None):
        loss = torch.mean(losses)
        loss.backward()


def make_mtl_weighter(cfgs, device, model_loss, epoch=None):
    if cfgs.weighter == 'DB_MTL':
        mtl_weighter = DB_MTL_v1(cfgs, model_loss, device)
    elif cfgs.weighter == 'Aligned_MTL':
        mtl_weighter = Aligned_MTL_v1(cfgs, model_loss, device)
    elif cfgs.weighter == 'MoCo_MTL':
        mtl_weighter = MoCo_MTL_v1(cfgs, model_loss, device)
    elif cfgs.weighter == 'STCH_MTL':
        mtl_weighter = STCH_MTL_v1(cfgs, model_loss, device)
    elif cfgs.weighter == 'None':
        mtl_weighter = None_MTL(cfgs, model_loss, device)
    else:
        raise NotImplementedError
    return mtl_weighter