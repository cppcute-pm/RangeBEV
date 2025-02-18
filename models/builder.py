from .image_model import ImageNet, RenderNet
from .pointcloud_model import PCNet
from .crossmodal_model import CMVPR, CMVPRv2

def make_model(cfgs, device):
    if cfgs.modal_type == 1:
        model = ImageNet(cfgs.imgnet_cfgs, cfgs.out_dim)
    elif cfgs.modal_type == 2:
        model = PCNet(cfgs.pcnet_cfgs, cfgs.out_dim)
    elif cfgs.modal_type == 3:
        model = CMVPR(cfgs.cmvpr_cfgs, cfgs.out_dim)
    elif cfgs.modal_type == 4:
        model = RenderNet(cfgs.rendernet_cfgs, cfgs.out_dim)
    elif cfgs.modal_type == 5:
        model = CMVPRv2(cfgs.cmvpr2_cfgs, cfgs.out_dim)
    else:
        raise ValueError('model type not supported')
    model = model.to(device)
    return model