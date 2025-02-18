import torch
import torch.nn as nn
from .misc import get_world_size

def make_optimizer(cfgs, dataloader_cfgs, model, loss_fn):
    # params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #        continue
    #     params += [{"params": value}]
    batch_size = dataloader_cfgs.batch_sampler_cfgs.batch_size
    real_lr = cfgs.lr * batch_size * get_world_size()
    params = []
    for name, child in model.named_children():
        level_1_flag = True

        # for modal 1 and modal 2
        if "image_backbone_lr" in cfgs and name == "backbone":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.image_backbone_lr * batch_size * get_world_size()}]
        if "image_aggregator_lr" in cfgs and name == "aggregator":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.image_aggregator_lr * batch_size * get_world_size()}]
        if "pc_backbone_lr" in cfgs and name == "backbone":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.pc_backbone_lr * batch_size * get_world_size()}]
        if "pc_aggregator_lr" in cfgs and name == "aggregator":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.pc_aggregator_lr * batch_size * get_world_size()}]
        
        # for modal 3 and phase1
        if "image_encoder_lr" in cfgs and name == "image_encoder":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.image_encoder_lr * batch_size * get_world_size()}]
        if "pc_encoder_lr" in cfgs and name == "pc_encoder":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.pc_encoder_lr * batch_size * get_world_size()}]
        if "image_aggregator_lr" in cfgs and name == "image_aggregator":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.image_aggregator_lr * batch_size * get_world_size()}]
        if "pc_aggregator_lr" in cfgs and name == "pc_aggregator":
            level_1_flag = False
            for param in child.parameters():
                params += [{'params': param, 'lr': cfgs.pc_aggregator_lr * batch_size * get_world_size()}]
        

        if level_1_flag:
            for param in child.parameters():
                params += [{'params': param}]


    # params = [{'params': model.parameters()}]

    if isinstance(loss_fn, nn.Module):
        params.append({'params': loss_fn.parameters()})
    if cfgs.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=real_lr, weight_decay=cfgs.weight_decay)
    else:
        raise ValueError('Not supported optimizer type: {}'.format(cfgs.optimizer_type))
    
    return optimizer